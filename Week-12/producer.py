import json
import time
from confluent_kafka import Producer

config = {
    'bootstrap.servers':'localhost:9092'
}

producer = Producer(config)

def publish_transaction(user_id:str,amount:float,timestamp:float):
    """
    Serializes a transaction event as Json and publlishes it to kafka 
    """
    
    topic = "transactions"
    
    #DEFINING PAYLOAD 
    payload = {
        "user_id":user_id,
        "amount":  amount,
        "timestamp": timestamp
    }
    
    # SERIALIZe the key and value to bytes 
    serialized_key = str(user_id).encode('utf-8')
    serialized_value = json.dumps(payload).encode('utf-8')
    
    # PRodcue the message to the transcationtopic 
    producer.produce(
        topic=topic,
        key=serialized_key,
        value=serialized_value
    )
    
    # FLush to ensure the queue is clear and delivered immediately
    producer.flush()
    
# 2. Publish three test transactions for user u_4821
if __name__ == "__main__":
    now = time.time()
    
    print("Publishing test transactions to Kafka...")
    
    # Transaction 1: ₹850.0 (Right now)
    publish_transaction(user_id="u_4821", amount=850.0, timestamp=now)
    
    # Transaction 2: ₹200.0 (1 hour ago)
    publish_transaction(user_id="u_4821", amount=200.0, timestamp=now - 3600)
    
    # Transaction 3: ₹450.0 (2 hours ago)
    publish_transaction(user_id="u_4821", amount=450.0, timestamp=now - 7200)
    
    print("Successfully published 3 events to the 'transactions' topic.")