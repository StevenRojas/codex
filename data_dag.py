import redis
import json
import re
import os
import boto3
from openai import OpenAI
import torch
import numpy as np
from dateutil import parser
import uuid
import traceback
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from datetime import datetime, timedelta
from transformers import AutoTokenizer, AutoModel
from redis.commands.search.field import TextField, NumericField, VectorField, TagField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.exceptions import ResponseError
from opensearchpy import OpenSearch, RequestsHttpConnection
from pydantic import BaseModel, ConfigDict, validator, root_validator
from typing import Union, Optional, ClassVar
from airflow.utils.log.logging_mixin import LoggingMixin
from dateutil.parser import parse
from typing import Dict, Any

logger = LoggingMixin().log
# Configuration for Redis and OpenSearch
REDIS_HOST = os.getenv('REDIS_HOST', '')
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
REDIS_PASSWORD = os.getenv('REDIS_PASSWORD', '')
MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'

OPENSEARCH_HOST = os.getenv('OPENSEARCH_HOST', 'opensearch-dev')
OPENSEARCH_PORT = int(os.getenv('OPENSEARCH_PORT', 9200))
OPENSEARCH_USER = os.getenv('OPENSEARCH_USER', 'admin')
OPENSEARCH_PASSWORD = os.getenv('OPENSEARCH_PASSWORD', 'Password')

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID', '')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY', '')
AWS_REGION = os.getenv('AWS_REGION', 'us-east-2')
SQS_QUEUE_URL = os.getenv('SQS_QUEUE_URL', 'https://sqs.us-east-2.amazonaws.com/0/sync_queue')

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 6, 14),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=1),
}


CATEGORIES = [
    'technical_support',
    'general_information',
    'feedback',
    'product_inquiry',
    'account_management',
    'sales_inquiry'
]


def classify_tag_to_category(tag_input: str) -> str:
    """
    Classify a given tag name into its corresponding category based on a predefined mapping.

    Parameters:
        tag (str): The tag name to classify.

    Returns:
        str: The relevant category if the tag is found, otherwise 'Uncategorized'.
    """
    # Mapping of Tag Names to their Relevant Categories.
    tag_to_id = {
        # Visual & Multimedia (Category 9)
        "Generative": 9,
        "Image Imp": 10,
        "Image Gen": 92,
        "Image Edit": 93,
        "Video Gen": 94,
        "Video Edit": 95,
        "3D Model": 96,
        "Computer": 97,  # Computer Vision
        "Visual Rec": 98,  # Visual Recognition
        "Graphic D": 99,  # Graphic Design
        "Animator": 100,
        "Augmente": 101,  # Augmented Reality
        "Virtual Reality": 102,
        "Photograph": 103,
        "Filmmaker": 104,
        "Digital Art": 105,
        "Translate": 109,
        "Summarize": 110,
        
        # Entertainment (Category 2)
        "Games": 57,
        "Interactive": 58,  # Interactive Fiction
        "AI Character": 59,
        "Roleplayin": 60,  # Roleplaying
        "Creative W": 61,  # Creative Writing
        "Humorous": 62,
        "Personally": 63,  # Personalized Recommendations
        "Social Media": 64,
        "NSFW (for": 65,
        "Story Telling": 66,
        "World Build": 67,  # World Building
        
        # Computation & Analysis (Category 12)
        "Machine Learning": 68,  # Machine Learning
        "Deep Learning": 69,  # Deep Learning
        "Data Analytics": 70,
        "Data Visualization": 71,  # Data Visualization
        "Predictive Modeling": 72,  # Predictive Modeling
        "Statistical Analysis": 73,  # Statistical Analysis
        "Mathematics": 74,  # Mathematics
        "Algorithm Design": 75,  # Algorithm Design
        "Simulation": 76,
        "Financial Modeling": 77,  # Financial Modeling
        "Business Intelligence": 78,  # Business Intelligence
        "Big Data": 79,
        "Cloud Computing": 80,  # Cloud Computing
        "Robotics": 81,
        
        # Audio & Speech (Category 11)
        "Speech Recognition": 82,  # Speech Recognition
        "Speech Synthesis": 83,  # Speech Synthesis
        "Voice Cloning": 84,  # Voice Cloning
        "Music Generation": 85,
        "Audio Edit": 86,  # Audio Editing
        "Sound Effects": 87,
        "Voice Assistants": 88,  # Voice Assistants
        "Podcast Creation": 89,  # Podcast Creation
        "Transcription": 90,  # Transcription
        
        # Text & Language (Category 10)
        "Natural Language": 106,  # Natural Language Processing
        "Large Language": 108,  # Large Language Models
        "Chatbots": 111,
        "Writing Assistant": 112,  # Writing Assistant
        "Content Creation": 113,  # Content Creation
        "Code Generation": 114,  # Code Generation
        "Fiction & Poetry Generation": 115,  # Fiction & Poetry Generation
        "Script Writing": 116,  # Script Writing
        "Text Classification": 118,  # Text Classification
        "Knowledge Base": 119,  # Knowledge Base
        "SEO & Marketing": 120,
        "Customer Service": 121,  # Customer Service
        "Developer Tools": 122  # Developer Tools
    }

    # Extract text to classify from either string or document
    if isinstance(tag_input, str):
        text_to_classify = tag_input
    elif isinstance(tag_input, FlexibleDocument):
        # Try to get meaningful text from document based on its type
        if hasattr(tag_input, 'message'):
            text_to_classify = tag_input.message
        elif hasattr(tag_input, 'title'):
            text_to_classify = tag_input.title
        elif hasattr(tag_input, 'name'):
            text_to_classify = tag_input.name
        elif hasattr(tag_input, 'summary'):
            text_to_classify = tag_input.summary
        else:
            text_to_classify = str(tag_input)
    else:
        text_to_classify = str(tag_input)

    logger.info(f"Parsed response to classify: {text_to_classify}")

    try:
        client = OpenAI(api_key=OPENAI_API_KEY)

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": f"""
                    Analyze the following content and identify the top 3 most relevant categories based on the provided list:

                    {json.dumps(tag_to_id, indent=2)}

                    Your task:
                    - Evaluate the content's relevance to each category.
                    - Only include categories with a confidence level of **at least 70%**.
                    - Return **a JSON array of integer IDs**, ordered by relevance (most relevant first).
                    - If **no category meets** the 70% relevance threshold, return an **empty array**.

                    If the content appears too generic, vague, or meaningless (e.g., short texts like "test", "hello", "random"), assume it's **not classifiable** and return an empty array.

                    Example:
                    Content: "An in-depth guide to prompt engineering and fine-tuning LLMs"
                    Output: [57, 9, 12]

                    Content to classify:
                    {text_to_classify}
                    """
                }

            ],
            response_format={"type": "json_object"},
            max_tokens=50,
            temperature=0.3
        )

        logger.info(f"openai response: {response.choices[0].message.content}")

        result = json.loads(response.choices[0].message.content.strip())

        logger.info(f"Parsed openai JSON: {result['result']}")

        if "result" in result:
            return result['result']
        else:
            return result['ids']

    except json.JSONDecodeError:
            logger.error(f"Failed to parse OpenAI response: {result}")
            return []

    except Exception as e:
        logger.error(f"Error in OpenAI classification: {str(e)}")
        # Fallback to simple text matching
        clean_text = text_to_classify.lower()
        matched_ids = [
            v for k, v in tag_to_id.items() 
            if k.lower() in clean_text
        ]
        return matched_ids[:3] or []


def string_to_tuple(s: str) -> tuple:
    if not s:
        return ("", "")

    s = s.strip().strip("()")
    parts = [part.strip() for part in s.split(",")]

    if len(parts) == 2:
        return tuple(parts)
    elif len(parts) == 1:
        return (parts[0], "")
    else:
        return ("", "")


class FlexibleDocument(BaseModel):
    id: Union[int, str] = None  # Allow id to be set dynamically based on content
    published: int = 0
    pricingModel: Optional[str] = ""  # Define pricing_model field to avoid validation errors
    model_config = ConfigDict(extra='allow')

    # UUID pattern for checking if the ID is a UUID
    UUID_PATTERN: ClassVar[re.Pattern] = re.compile(
        r'^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$'
    )

    @root_validator(pre=True)
    def set_id_from_post_id(cls, values):
        # Determine id based on 'id' or 'post_id' presence
        id_value = values.get('id') or values.get('postId')

        if id_value is None:
            id_value = str(uuid.uuid4())

        # Check if id_value is a UUID; if so, keep it as a string
        if isinstance(id_value, str) and cls.UUID_PATTERN.match(id_value):
            values['id'] = id_value
        else:
            # Try converting to int if it's a numeric value
            try:
                values['id'] = int(id_value)
            except ValueError:
                values['id'] = id_value  # Keep as string if non-numeric and non-UUID

        return values

    @validator('published', pre=True, always=True)
    def set_published(cls, v, values):
        publish_value = values.get('publish', v)
        if isinstance(publish_value, bool):
            return int(publish_value)
        if isinstance(publish_value, str):
            if publish_value.lower() == 'true':
                return 1
            elif publish_value.lower() == 'false':
                return 0
        return int(publish_value) if publish_value is not None else 0

    @validator('pricingModel', pre=True, always=True)
    def validate_pricing_model(cls, v):
        return str(v) if v is not None else v


os_client = OpenSearch(
    hosts=[{
        'host': OPENSEARCH_HOST,
        'port': OPENSEARCH_PORT,
    }],
    http_auth=(OPENSEARCH_USER, OPENSEARCH_PASSWORD),
    use_ssl=False,
    verify_certs=False,
    connection_class=RequestsHttpConnection,
)


def classify_document(text: str) -> str:
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        # Send the content to OpenAI's model for classification
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                    {
                        "role": "user",
                        "content": f"""
                        Classify the intent of the input text into the following categories: {', '.join(CATEGORIES)}. 
                        Return only one category:

                        Always follow this format:
                        category_1

                        Input text:
                        {text}

                        Category: (only return the category nothing else)
                        """
                            }
                    ],
            max_tokens=20,
            n=1,
            stop=None,
            temperature=0.5
        )

        category = response.choices[0].message.content
        return category
    except Exception as e:
        logger.info(f"Error classifying document: {e}")
        return {'general_information': 0.90}


def remove_html_tags(text):
    """Remove HTML tags from a string."""
    if text is None:
        return ""
    clean = re.compile(r'<.*?>')
    return re.sub(clean, '', text)


def save_to_postgres(document, index):
    """
    Save specific document fields into Postgres based on the index type.
    For:
      - ai_news    -> news_metadata table
      - ai_post    -> posts_metadata table
      - ai_tools   -> product_metadata table
    """
    try:
        # Instantiate PostgresHook with explicit connection details.
        pg_hook = PostgresHook(postgres_conn_id='postgres_robot_cache')
        conn = pg_hook.get_conn()
        cursor = conn.cursor()
        
        # Remove binary data that can't be JSON serialized if we need to dump later
        dump_document = {k: v for k, v in document.items() if not isinstance(v, bytes)}
        
        relevant_category = document.get("relevant_category") or document.get("relevantCategory")

        if relevant_category:
            s_clean = relevant_category.strip("[]").strip()
            relevant_category_arr = [
                cat.strip().strip("'").strip('"')
                for cat in s_clean.split(",")
                if cat.strip()
            ]
            relevant_category_str = ",".join(relevant_category_arr)
        else:
            relevant_category_str = ""

        pg_id = None  # Will store the PostgreSQL-generated ID

        if index == 'ai_news':
            # Extract fields for news_metadata table
            title = remove_html_tags(document.get("title"))
            summary = remove_html_tags(document.get("summary"))
            source = remove_html_tags(document.get("source"))
            link = remove_html_tags(document.get("link"))
            news_id = document.get("id")
            raw_date = document.get("publishDate")

            if raw_date:
                try:
                    if isinstance(raw_date, int):
                        date_published = datetime.fromtimestamp(raw_date)
                    else:
                        date_published = parser.parse(raw_date)
                except Exception as parse_err:
                    logger.error(f"Error parsing date_published: {parse_err}")
                    date_published = datetime.now()
            else:
                date_published = datetime.now()

            category = remove_html_tags(document.get("category", ""))
            categories = document.get("categories", [])
            if isinstance(categories, str):
                try:
                    parsed_categories = json.loads(categories)
                    if isinstance(parsed_categories, list):
                        categories = parsed_categories
                    else:
                        categories = [str(parsed_categories)]
                except Exception:
                    categories = [c.strip() for c in categories.split(",") if c.strip()]
            
            category_names = []
            for cat in categories:
                if isinstance(cat, dict) and "name" in cat:
                    category_names.append(cat["name"])
                else:
                    category_names.append(str(cat))
            categories_str = ", ".join(category_names) if category_names else ""
            user_id = document.get("user_id", "0")

            sql = """
                INSERT INTO news_metadata 
                    (news_id, title, summary, source, link, date_published, categories, category_generated, user_id, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW())
                ON CONFLICT (news_id) DO UPDATE SET
                    title = EXCLUDED.title,
                    summary = EXCLUDED.summary,
                    source = EXCLUDED.source,
                    link = EXCLUDED.link,
                    date_published = EXCLUDED.date_published,
                    categories = EXCLUDED.categories,
                    category_generated = EXCLUDED.category_generated,
                    user_id = EXCLUDED.user_id,
                    updated_at = NOW()
                RETURNING id;
            """
            params = (
                news_id,
                title,
                summary,
                source,
                link,
                str(date_published),
                relevant_category_str,
                category,
                user_id
            )

        elif index == 'ai_post':
            # Extract fields for posts_metadata table
            category = remove_html_tags(document.get("category"))
            id = document.get("id")
            post_id = document.get("postId")
            comments = document.get("comments", 0)

            try:
                comments = int(comments) if comments not in (None, '') else 0
            except (ValueError, TypeError):
                comments = 0

            likes = document.get("likes", 0)
            try:
                likes = int(likes) if likes not in (None, '') else 0
            except (ValueError, TypeError):
                likes = 0

            message = remove_html_tags(document.get("message"))
            post_type = remove_html_tags(document.get("postType") or document.get("post_type"))
            user_id = document.get("userId", '0')

            sql = """
                INSERT INTO post_metadata 
                    (id, post_id, comments, likes, message, post_type, author_id, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, NOW(), NOW())
                ON CONFLICT (post_id) DO UPDATE SET
                    post_id = EXCLUDED.post_id,
                    comments = EXCLUDED.comments,
                    likes = EXCLUDED.likes,
                    message = EXCLUDED.message,
                    post_type = EXCLUDED.post_type,
                    author_id = EXCLUDED.author_id,
                    updated_at = NOW()
                RETURNING id;
            """
            params = (
                id,
                post_id,
                comments,
                likes,
                message,
                post_type,
                user_id
            )

        elif index == 'ai_tools':
            # Extract fields for product_metadata table
            prod_id = document.get("id")
            name = remove_html_tags(document.get("name"))
            reference = remove_html_tags(document.get("reference"))
            price_model = remove_html_tags(document.get("pricingModel") or document.get("price_model"))
            short_description = remove_html_tags(document.get("description") or document.get("shortDescription"))
            long_description = remove_html_tags(document.get("longDescription") or document.get("long_description"))
            category = remove_html_tags(document.get("category"))
            keywords = document.get("keywords", [])

            if isinstance(keywords, str):
                keywords = keywords.replace('\\"', '"')
                try:
                    keywords = json.loads(keywords)
                except Exception:
                    keywords = [k.strip() for k in keywords.split(",") if k.strip()]
            elif isinstance(keywords, list):
                keywords = [keyword.replace('\\"', '"') for keyword in keywords]

            keywords_str = ", ".join(keywords) if keywords else ""

            sql = """
                INSERT INTO product_metadata
                    (product_id, name, reference, price_model, short_description, long_description, category_generated, keywords, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW())
                ON CONFLICT (product_id) DO UPDATE SET
                    name = EXCLUDED.name,
                    reference = EXCLUDED.reference,
                    price_model = EXCLUDED.price_model,
                    short_description = EXCLUDED.short_description,
                    long_description = EXCLUDED.long_description,
                    category_generated = EXCLUDED.category_generated,
                    keywords = EXCLUDED.keywords,
                    updated_at = NOW()
                RETURNING id;
            """
            params = (
                prod_id,
                name,
                reference,
                price_model,
                short_description,
                long_description,
                category,
                keywords_str
            )
        else:
            logger.info(f"Index {index} does not require Postgres sync.")
            return None

        cursor.execute(sql, params)
        pg_id = cursor.fetchone()[0]
        conn.commit()
        
        if index == 'ai_news' and relevant_category_str:
            tag_ids = [int(tag_id.strip()) for tag_id in relevant_category_str.split(',') if tag_id.strip()]
            update_news_tags(pg_id, tag_ids)
        elif index == 'ai_post' and relevant_category_str:
            tag_ids = [int(tag_id.strip()) for tag_id in relevant_category_str.split(',') if tag_id.strip()]
            update_post_tags(pg_id, tag_ids)

        logger.info(f"Document saved to Postgres for index {index} with PG ID {pg_id}.")
        return pg_id

    except Exception as e:
        logger.error(f"Error saving document to Postgres: {e}")
        logger.error(traceback.format_exc())
        save_to_dump_file(dump_document)
        return None
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()


def process_event(**kwargs):
    message_body = kwargs['dag_run'].conf.get('message_body')
    logger.info(f"Received message body: {message_body}")

    try:
        data = json.loads(message_body.strip())
        logger.info(f"Parsed JSON: {data}")
    except json.JSONDecodeError as e:
        logger.info(f"Error parsing JSON: {e} - Message body: {message_body}")
        return

    action = data.get("action", "").lower()

    if action == "save":
        save_document(data)
    elif action == "delete":
        delete_document(data)
    else:
        logger.info(f"Unknown action: {action}")


def _parse_keywords(raw_keywords: Any) -> str:
    """Normalize keyword data into a comma-separated string."""
    if isinstance(raw_keywords, str):
        raw_keywords = raw_keywords.replace('\\"', '"')
        try:
            raw_keywords = json.loads(raw_keywords)
        except Exception:
            raw_keywords = [k.strip() for k in raw_keywords.split(',') if k.strip()]

    if isinstance(raw_keywords, list):
        raw_keywords = [k.strip() for k in raw_keywords if isinstance(k, str)]
        return ', '.join(raw_keywords)

    return str(raw_keywords) if raw_keywords else ''


def _prepare_document(data: dict, index: str) -> FlexibleDocument:
    """Prepare and normalize incoming document data."""
    if index == 'ai_news':
        data.setdefault('topic', '')
        data.setdefault('relevant_category', '')
    elif index == 'ai_post':
        data.setdefault('topic', '')
        data.setdefault('relevantCategory', '')
    elif index == 'ai_tools':
        data['longDescription'] = data.get('description', '')
        if 'icon' in data:
            data['icon'] = f"https://cdn.cloudhands.ai/{data['icon']}"

    document = FlexibleDocument(**data)

    if index == 'ai_tools':
        document.keywords = _parse_keywords(data.get('keywords'))

    document.category = str(classify_document(str(data)))
    return document


def _ensure_redis_index(client: redis.Redis, index: str, doc: FlexibleDocument) -> None:
    """Ensure the Redis search index exists."""
    try:
        client.ft(index).info()
        return
    except ResponseError:
        pass

    schema = [
        TextField("category"),
        NumericField("published"),
        VectorField(
            "embedding",
            "FLAT",
            {"TYPE": "FLOAT32", "DIM": 384, "DISTANCE_METRIC": "COSINE"},
        ),
    ]

    for key, value in doc.dict().items():
        if extract_timestamp(value) is not None:
            schema.append(NumericField(key))

    if index == 'ai_tools':
        schema = [
            TextField("name"),
            TextField("category"),
            NumericField("published"),
            VectorField("embedding", "FLAT", {"TYPE": "FLOAT32", "DIM": 384, "DISTANCE_METRIC": "COSINE"}),
        ]
    elif index == 'ai_post':
        schema.extend([
            TextField("postType"),
            TextField("messageType"),
            TagField("languages"),
            TextField("relevantCategory"),
        ])
    elif index == 'ai_news':
        schema.extend([
            TextField("source"),
            TextField("type"),
            TextField("topic"),
            TextField("relevant_category"),
        ])

    prefix = f"{index}:"
    client.ft(index).create_index(
        fields=schema,
        definition=IndexDefinition(prefix=[prefix], index_type=IndexType.HASH),
    )


def _generate_embedding(text: str) -> tuple[np.ndarray | None, bytes | None]:
    """Generate a vector embedding from text."""
    if not text:
        return None, None

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    with torch.no_grad():
        embedding = model(**inputs).last_hidden_state.mean(dim=1).squeeze().numpy()

    return embedding, np.array(embedding, dtype=np.float32).tobytes()



def save_document(data):
    """Refactored document save handler."""
    client = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        password=REDIS_PASSWORD,
        decode_responses=False,
    )

    try:
        index = data.get("index", "default")

        if index == "ai_tags":
            save_tags_to_postgres(data, index)
            return

        document_data = _prepare_document(data["data"], index)
        key = f"{index}:{document_data.id}"

        _ensure_redis_index(client, index, document_data)

        if index == "ai_news":
            fields = [document_data.title, document_data.summary]
        elif index == "ai_post":
            fields = [document_data.message or "no content"]
        elif index == "ai_tools":
            fields = [document_data.name, document_data.longDescription]
        else:
            fields = []

        combined_text = " ".join(filter(None, fields))
        embedding, embedding_bytes = _generate_embedding(combined_text)

        document = document_data.model_dump(by_alias=True)
        if index != "ai_tools":
            document.pop("pricingModel", None)
        if embedding_bytes:
            document["embedding"] = embedding_bytes

        for k, v in document.items():
            timestamp = extract_timestamp(v)
            if timestamp is not None:
                document[k] = timestamp
            if isinstance(v, (dict, list)):
                document[k] = json.dumps(v)
            elif v is None:
                document[k] = ""
            elif isinstance(v, bool):
                document[k] = int(v)

        client.hset(name=key, mapping=document)
        client.sadd(f"{index}:_keys", key)
        logger.info(f"Document with ID {document_data.id} saved in Redis under index '{index}'.")

        if index in ['ai_news', 'ai_post', 'ai_tools']:
            save_to_postgres(document, index)
        if index in ['ai_post', 'ai_news']:
            send_to_sqs(document, index)
        if embedding is not None:
            opensearch_doc = document.copy()
            opensearch_doc['embedding'] = embedding.tolist()
            save_to_opensearch(opensearch_doc, index)
    except Exception as e:
        logger.info(f"Error processing document ID {data['data'].get('id')}: {e}")
        logger.info(traceback.format_exc())
        save_to_dump_file(data['data'])


def save_to_opensearch(document, index):
    try:

        new_id = str(uuid.uuid4())

        if index == 'ai_news':
            document['ai_news_id'] = document['id']
        
        if index == 'ai_post':
            document['ai_post_id'] = document['id']

        if index == 'ai_tools':
            document['ai_tools_id'] = document['id']

        document['id'] = new_id

        # Ensure the 'type' field exists in the document before saving to OpenSearch
        if 'type' not in document:
            document['type'] = 'suggestions'  # Add default type if missing

        # Check if the document exists in OpenSearch
        if os_client.exists(index=document['type'], id=str(document['id'])):
            logger.info(f"Updating document with ID {str(document['id'])} in OpenSearch.")
        else:
            logger.info(f"Creating document with ID {str(document['id'])} in OpenSearch.")

        # Index or update the document in OpenSearch
        response = os_client.index(
            index=document['type'],
            body=document,
            id=str(document['id'])
        )
        logger.info(f"Document with ID {str(document['id'])} indexed in OpenSearch: {response}")

    except Exception as e:
        logger.info(f"Error saving document to OpenSearch: {e}")
        save_to_dump_file(document)


def delete_document(data):

    client = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        password=REDIS_PASSWORD,
        decode_responses=True
    )

    index = data.get("index", "default")
    key = f"{index}:{data['id']}"

    if index == 'ai_post' and data['data'].get('amid', False):
        data['data']['admin_id'] = data['data'].get('user_id')

    # Delete from Redis
    if not client.exists(key):
        logger.info(f"Document with ID {data['id']} does not exist in Redis index '{index}'. Cannot delete.")
    else:
        client.delete(key)
        client.srem(f"{index}:_keys", key)
        logger.info(f"Document with ID {data['id']} deleted from Redis index '{index}'.")

    # Delete from OpenSearch
    try:
        if os_client.exists(index=index, id=data['id']):
            os_client.delete(index=index, id=data['id'])
            logger.info(f"Document with ID {data['id']} deleted from OpenSearch.")
        else:
            logger.info(f"Document with ID {data['id']} does not exist in OpenSearch index '{index}'. Cannot delete.")

    except Exception as e:
        logger.info(f"Error deleting document from OpenSearch: {e}")


def extract_timestamp(value):
    """
    Check if the given value can be interpreted as a date.
    If yes, return its Unix timestamp; otherwise, return None.
    """
    try:
        if isinstance(value, str):
            # Use dateutil.parser to handle complex and ISO 8601 date formats
            parsed_date = parse(value)  # Automatically detects and parses ISO 8601 formats
            return int(parsed_date.timestamp())  # Convert to Unix timestamp
        elif isinstance(value, datetime):
            return int(value.timestamp())  # Direct conversion if already a datetime object
        return None
    except Exception:
        return None


def send_to_sqs(document: Dict[str, Any], index: str) -> None:
    """
    Send categorized document to AWS SQS queue after processing.

    Parameters:
        document (dict): The processed document data
        index (str): The document type ('ai_post' or 'ai_news')
    """
    try:
        # Initialize SQS client
        sqs = boto3.client(
            'sqs',
            region_name=AWS_REGION,
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY
        )

        queue_url = SQS_QUEUE_URL

        del document['embedding']

        if not queue_url:
            logger.info(f"No SQS queue configured for index {index}")
            return

        logger.info(f" SQS document {document}")
        # Prepare message attributes (optional)
        message_attributes = {
            'DocumentType': {
                'DataType': 'String',
                'StringValue': index
            },
            'DocumentId': {
                'DataType': 'String',
                'StringValue': str(document.get('id', ''))
            }
        }

        relevant_category = document.get("relevant_category") or document.get("relevantCategory")

        if relevant_category:
            s_clean = relevant_category.strip("[]").strip()
            relevant_category_arr = [
                int(cat.strip().strip("'").strip('"'))
                for cat in s_clean.split(",")
                if cat.strip()
            ]
        else:
            relevant_category_arr = []

        logger.info(f" Message id {document.get('id')}")
        message = {
            "id": document.get("id"),
            "tags": relevant_category_arr,
            "type": "news" if index == "ai_news" else "post"
        }

        logger.info(f" Meesage SQS document {message}")
        # Send message to SQS
        response = sqs.send_message(
            QueueUrl=queue_url,
            MessageBody=json.dumps(message),
            MessageAttributes=message_attributes
        )

        logger.info(f"Successfully sent document {document.get('id')} to SQS {index} queue. MessageId: {response['MessageId']}")

    except Exception as e:
        logger.error(f"Error sending document to SQS: {e}")
        logger.error(traceback.format_exc())


def save_to_dump_file(data):
    dump_file = "failed_ingestions.json"

    # Check if the file already exists
    if os.path.exists(dump_file):
        # If it exists, read the existing content
        with open(dump_file, "r") as f:
            try:
                failed_data = json.load(f)
            except json.JSONDecodeError:
                failed_data = []
    else:
        failed_data = []

    # Add the failed document data to the list
    failed_data.append(data)

    # Write the updated list back to the file
    with open(dump_file, "w") as f:
        json.dump(failed_data, f, indent=4)

    logger.info(f"Document ID {data['id']} saved to dump file {dump_file} due to failure.")


def save_tags_to_postgres(document, index):
    """
    Save or update tags in PostgreSQL for the ai_tags index.
    Creates the tags table and junction tables if they don't exist.
    """
    try:
        pg_hook = PostgresHook(postgres_conn_id='postgres_robot_cache')
        conn = pg_hook.get_conn()
        cursor = conn.cursor()

        # Create tags table if it doesn't exist
        create_tags_table_sql = """
        CREATE TABLE IF NOT EXISTS tags_metadata (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            category_id INTEGER NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );
        """
        cursor.execute(create_tags_table_sql)

        # Create junction table for post-tag relationships if it doesn't exist
        create_post_tags_table_sql = """
        CREATE TABLE IF NOT EXISTS post_tags (
            post_id INTEGER,
            tag_id INTEGER,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            PRIMARY KEY (post_id, tag_id),
            FOREIGN KEY (post_id) REFERENCES post_metadata (id) ON DELETE CASCADE,
            FOREIGN KEY (tag_id) REFERENCES tags_metadata (id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_post_tags_tag_id ON post_tags(tag_id);
        """
        cursor.execute(create_post_tags_table_sql)

        # Create junction table for news-tag relationships if it doesn't exist
        create_news_tags_table_sql = """
        CREATE TABLE IF NOT EXISTS news_tags (
            news_id INTEGER,
            tag_id INTEGER,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            PRIMARY KEY (news_id, tag_id),
            FOREIGN KEY (news_id) REFERENCES news_metadata (id) ON DELETE CASCADE,
            FOREIGN KEY (tag_id) REFERENCES tags_metadata (id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_news_tags_tag_id ON news_tags(tag_id);
        """
        cursor.execute(create_news_tags_table_sql)

        # Create trigger for update timestamp if it doesn't exist
        trigger_sql = """
        CREATE OR REPLACE FUNCTION update_timestamp()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = NOW();
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
        
        DO $$
        BEGIN
            IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'update_tags_timestamp') THEN
                CREATE TRIGGER update_tags_timestamp
                BEFORE UPDATE ON tags_metadata
                FOR EACH ROW EXECUTE FUNCTION update_timestamp();
            END IF;
        END $$;
        """
        cursor.execute(trigger_sql)
        conn.commit()
        
        # Prepare upsert query for tags
        upsert_tag_sql = """
        INSERT INTO tags_metadata (id, name, category_id)
        VALUES (%s, %s, %s)
        ON CONFLICT (id) DO UPDATE SET
            name = EXCLUDED.name,
            category_id = EXCLUDED.category_id;
        """
        
        # Process each tag
        tags = document.get('data', [])
        for tag in tags:
            tag_id = tag.get('id')
            name = tag.get('name')
            category_id = tag.get('categoryId')
            
            if None in (tag_id, name, category_id):
                logger.info(f"Skipping incomplete tag: {tag}")
                continue
                
            cursor.execute(upsert_tag_sql, (tag_id, name, category_id))
        
        conn.commit()
        logger.info(f"Successfully processed {len(tags)} tags for index {index}")
        
    except Exception as e:
        logger.error(f"Error saving tags to PostgreSQL: {e}")
        logger.error(traceback.format_exc())
        if conn:
            conn.rollback()
        save_to_dump_file(document)
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def update_post_tags(id: int, tag_ids: list[int]):
    """Update the many-to-many relationship between a post and tags"""
    try:
        pg_hook = PostgresHook(postgres_conn_id='postgres_robot_cache')
        conn = pg_hook.get_conn()
        cursor = conn.cursor()
        
        # First delete existing relationships for this post
        delete_sql = "DELETE FROM post_tags WHERE post_id = %s"
        cursor.execute(delete_sql, (id,))
        
        # Insert new relationships
        if tag_ids:
            insert_sql = "INSERT INTO post_tags (post_id, tag_id) VALUES (%s, %s)"
            cursor.executemany(insert_sql, [(id, tag_id) for tag_id in tag_ids])
        
        conn.commit()
        logger.info(f"Updated tags for post {id} with {len(tag_ids)} tags")
        
    except Exception as e:
        logger.error(f"Error updating post tags: {e}")
        if conn:
            conn.rollback()
        raise
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def update_news_tags(news_id: int, tag_ids: list[int]):
    """Update the many-to-many relationship between a news item and tags"""
    try:
        pg_hook = PostgresHook(postgres_conn_id='postgres_robot_cache')
        conn = pg_hook.get_conn()
        cursor = conn.cursor()
        
        # First delete existing relationships for this news item
        delete_sql = "DELETE FROM news_tags WHERE news_id = %s"
        cursor.execute(delete_sql, (news_id,))
        
        # Insert new relationships
        if tag_ids:
            insert_sql = "INSERT INTO news_tags (news_id, tag_id) VALUES (%s, %s)"
            cursor.executemany(insert_sql, [(news_id, tag_id) for tag_id in tag_ids])
        
        conn.commit()
        logger.info(f"Updated tags for news {news_id} with {len(tag_ids)} tags")
        
    except Exception as e:
        logger.error(f"Error updating news tags: {e}")
        if conn:
            conn.rollback()
        raise
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def create_dag() -> DAG:
    with DAG(
        dag_id='sqs_processor',
        default_args=default_args,
        description='Process messages from Lambda-triggered SQS events',
        schedule_interval=None,
        catchup=False,
        max_active_runs=40,
    ) as dag:
        PythonOperator(
            task_id='process_event',
            provide_context=True,
            python_callable=process_event,
        )
    return dag


dag = create_dag()
