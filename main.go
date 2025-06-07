package main

import (
	"context"
	"database/sql"
	"encoding/json"
	"log"
	"os"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/sqs"
	sqstypes "github.com/aws/aws-sdk-go-v2/service/sqs/types"
	_ "github.com/lib/pq"
	opensearch "github.com/opensearch-project/opensearch-go/v2"
	"github.com/redis/go-redis/v9"
)

type Message struct {
	Action string          `json:"action"`
	Index  string          `json:"index"`
	Data   json.RawMessage `json:"data"`
}

func main() {
	queueURL := os.Getenv("SQS_QUEUE_URL")
	if queueURL == "" {
		log.Fatal("SQS_QUEUE_URL must be set")
	}

	ctx := context.Background()

	// Initialize connections for repository
	redisAddr := os.Getenv("REDIS_ADDR")
	redisPassword := os.Getenv("REDIS_PASSWORD")
	rdb := redis.NewClient(&redis.Options{Addr: redisAddr, Password: redisPassword})

	pgDSN := os.Getenv("POSTGRES_DSN")
	db, err := sql.Open("postgres", pgDSN)
	if err != nil {
		log.Fatalf("postgres open: %v", err)
	}

	osAddr := os.Getenv("OPENSEARCH_ADDR")
	osUser := os.Getenv("OPENSEARCH_USER")
	osPass := os.Getenv("OPENSEARCH_PASSWORD")
	osClient, err := opensearch.NewClient(opensearch.Config{Addresses: []string{osAddr}, Username: osUser, Password: osPass})
	if err != nil {
		log.Fatalf("opensearch client: %v", err)
	}

	repo := NewRepository(rdb, db, osClient)

	cfg, err := config.LoadDefaultConfig(ctx)
	if err != nil {
		log.Fatalf("unable to load AWS config: %v", err)
	}

	client := sqs.NewFromConfig(cfg)

	for {
		if err := pollOnce(ctx, client, repo, queueURL); err != nil {
			log.Printf("error polling messages: %v", err)
		}
		time.Sleep(5 * time.Second)
	}
}

func pollOnce(ctx context.Context, client *sqs.Client, repo Repository, queueURL string) error {
	out, err := client.ReceiveMessage(ctx, &sqs.ReceiveMessageInput{
		QueueUrl:            aws.String(queueURL),
		MaxNumberOfMessages: 10,
		WaitTimeSeconds:     20,
	})
	if err != nil {
		return err
	}
	for _, m := range out.Messages {
		if err := processMessage(ctx, repo, m); err != nil {
			log.Printf("failed to process message %s: %v", aws.ToString(m.MessageId), err)
			continue
		}
		_, err := client.DeleteMessage(ctx, &sqs.DeleteMessageInput{
			QueueUrl:      aws.String(queueURL),
			ReceiptHandle: m.ReceiptHandle,
		})
		if err != nil {
			log.Printf("failed to delete message %s: %v", aws.ToString(m.MessageId), err)
		}
	}
	return nil
}

func processMessage(ctx context.Context, repo Repository, m sqstypes.Message) error {
	var msg Message
	if err := json.Unmarshal([]byte(aws.ToString(m.Body)), &msg); err != nil {
		return err
	}
	log.Printf("received action=%s index=%s", msg.Action, msg.Index)
	switch msg.Action {
	case "save":
		return handleSave(ctx, repo, msg)
	case "delete":
		return handleDelete(ctx, repo, msg)
	default:
		log.Printf("unknown action: %s", msg.Action)
	}
	return nil
}

func handleSave(ctx context.Context, repo Repository, msg Message) error {
	var doc map[string]interface{}
	if err := json.Unmarshal(msg.Data, &doc); err != nil {
		return err
	}
	if err := repo.Save(ctx, msg.Index, doc); err != nil {
		return err
	}
	log.Printf("saved document %v in index %s", doc["id"], msg.Index)
	return nil
}

func handleDelete(ctx context.Context, repo Repository, msg Message) error {
	var payload struct {
		ID string `json:"id"`
	}
	if err := json.Unmarshal(msg.Data, &payload); err != nil {
		return err
	}
	if err := repo.Delete(ctx, msg.Index, payload.ID); err != nil {
		return err
	}
	log.Printf("deleted document %s from index %s", payload.ID, msg.Index)
	return nil
}
