import psycopg2
import redis 
import os 
from datetime import timezone,datetime

pg_conn = psycopg2.connect(os.environ.get("DATABASE_URL"))
redis_client =  redis.Redis.from_url(os.environ.get("REDIS_URL"))



with pg_conn.cursor() as cursor:
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS feature_store (
            user_id VARCHAR NOT NULL,
            feature_name VARCHAR NOT NULL,
            feature_value NUMERIC,
            computed_at TIMESTAMP WITH TIME ZONE NOT NULL,
            PRIMARY KEY (user_id, feature_name, computed_at)
        )
    """)
    pg_conn.commit()
    

def materialize_transaction(user_id:str,amount:float, timestamp: float):
    tx_key = f"user:{user_id}:transactions"
    hash_key = f"user:{user_id}"
    cutoff = timestamp - (7 * 24 * 60 * 60)
    
    member_data = f"{timestamp}:{amount}"
    
    pipeline = redis_client.pipeline()
    
    # 1. ZADD: Add the new transaction with its timestamp as the score 
    pipeline.zadd(tx_key, {member_data: timestamp})
    
    # 2. ZREMRANGEBYSCORE : Prune Everything older than 7 days 
    pipeline.zremrangebyscore(tx_key,'-inf',cutoff)
    
    # 3. ZRANGE : fetch all remaining members to compute the new average
    pipeline.zrange(tx_key,0,-1)
    results = pipeline.execute()
    
    valid_transactions = results[2]
    
    total_spend = 0.0
    for tx in valid_transactions:
        # REdis py returns bytes so we decode and extract the amount 
        tx_str = tx.decode('utf-8')
        _, tx_amount = tx_str.split(':')
        total_spend += float(tx_amount)
    
    tx_count = len(valid_transactions)
    new_average = total_spend / tx_count if tx_count > 0 else 0.0
    
    # Update the serving hash 
    redis_client.hset(hash_key,mapping={
        "7_day_average_spend": new_average,
        "time_since_last_transaction": 0
    })
    
    
materialize_transaction("u_4821", 850.0, 1700000000.0)
materialize_transaction("u_4821", 200.0, 1700000500.0)
materialize_transaction("u_4821", 450.0, 1700001000.0)

result = redis_client.hgetall("user:u_4821")
print(result)

def get_features_for_serving(user_id:str) -> dict:
    hash_key = f"user:{user_id}"
    
    # Fetch the entire hash from redis 
    raw_features = redis_client.hgetall(hash_key)
    
    if not raw_features:
        return {}
    
    features = {}
    for key,value in raw_features.items():
        clean_key = key.decode('utf-8')
        clean_value = float(value.decode('utf-8'))
        
        features[clean_key] = clean_value
        
    return features

print(get_features_for_serving("u_4821"))
print(get_features_for_serving("u_9999"))  # non-existent user


def write_to_offline_store(user_id:str,feature_name:str,feature_value:float, computed_at):
    # open cursor usingg a context manager for safe resource handling 
    with pg_conn.cursor() as cursor:
        
        # defining the parameterized query
        insert_query = """
            INSERT INTO feature_store (user_id,feature_name, feature_value,computed_at)
            VALUES (%s,%s,%s,%s)
            ON CONFLICT (user_id, feature_name, computed_at) 
            DO UPDATE SET feature_value = EXCLUDED.feature_value;
        """
        
        cursor.execute(
            insert_query,
            (user_id,feature_name,feature_value,computed_at)
        )
        
        pg_conn.commit()
        
result1 = write_to_offline_store("u_4821","7_day_average_spend",500.0 , datetime.now(timezone.utc))

print(result1)