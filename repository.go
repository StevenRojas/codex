package main

import (
	"bytes"
	"context"
	"database/sql"
	"encoding/json"
	"fmt"

	_ "github.com/lib/pq"
	opensearch "github.com/opensearch-project/opensearch-go/v2"
	"github.com/opensearch-project/opensearch-go/v2/opensearchapi"
	"github.com/redis/go-redis/v9"
)

// Repository defines persistence actions for documents.
type Repository interface {
	Save(ctx context.Context, index string, doc map[string]interface{}) error
	Delete(ctx context.Context, index, id string) error
}

type repo struct {
	redis *redis.Client
	db    *sql.DB
	os    *opensearch.Client
}

// NewRepository creates a Repository with the given connections.
func NewRepository(redis *redis.Client, db *sql.DB, os *opensearch.Client) Repository {
	return &repo{redis: redis, db: db, os: os}
}

func (r *repo) Save(ctx context.Context, index string, doc map[string]interface{}) error {
	id, ok := doc["id"].(string)
	if !ok {
		return fmt.Errorf("document missing id")
	}

	key := fmt.Sprintf("%s:%s", index, id)
	jsonData, err := json.Marshal(doc)
	if err != nil {
		return err
	}

	// Redis
	if err := r.redis.HSet(ctx, key, doc).Err(); err != nil {
		return fmt.Errorf("redis: %w", err)
	}
	if err := r.redis.SAdd(ctx, index+":_keys", key).Err(); err != nil {
		return fmt.Errorf("redis: %w", err)
	}

	// Postgres
	if r.db != nil {
		table := index + "_documents"
		_, err = r.db.ExecContext(ctx,
			fmt.Sprintf(`CREATE TABLE IF NOT EXISTS %s (id TEXT PRIMARY KEY, data JSONB)`, table))
		if err != nil {
			return fmt.Errorf("postgres create table: %w", err)
		}
		_, err = r.db.ExecContext(ctx,
			fmt.Sprintf(`INSERT INTO %s(id, data) VALUES($1,$2) ON CONFLICT(id) DO UPDATE SET data=excluded.data`, table),
			id, jsonData)
		if err != nil {
			return fmt.Errorf("postgres insert: %w", err)
		}
	}

	// OpenSearch
	if r.os != nil {
		req := opensearchapi.IndexRequest{
			Index:      index,
			DocumentID: id,
			Body:       bytes.NewReader(jsonData),
			Refresh:    "true",
		}
		if _, err := req.Do(ctx, r.os); err != nil {
			return fmt.Errorf("opensearch: %w", err)
		}
	}

	return nil
}

func (r *repo) Delete(ctx context.Context, index, id string) error {
	key := fmt.Sprintf("%s:%s", index, id)

	if err := r.redis.Del(ctx, key).Err(); err != nil {
		return fmt.Errorf("redis: %w", err)
	}
	r.redis.SRem(ctx, index+":_keys", key) // ignore error

	if r.db != nil {
		table := index + "_documents"
		_, err := r.db.ExecContext(ctx,
			fmt.Sprintf(`DELETE FROM %s WHERE id=$1`, table), id)
		if err != nil {
			return fmt.Errorf("postgres delete: %w", err)
		}
	}

	if r.os != nil {
		req := opensearchapi.DeleteRequest{Index: index, DocumentID: id}
		if _, err := req.Do(ctx, r.os); err != nil {
			return fmt.Errorf("opensearch: %w", err)
		}
	}

	return nil
}
