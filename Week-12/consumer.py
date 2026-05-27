import json 
from confluent_kafka import Consumer, KafkaError
from FeatureStore import materialize_transaction

config = {
    'bootstrap.servers' :  'localhost:9092',
    'group.id':'feature-store-workers',
    'auto.offset.reset':'earliest'
}

consumer = Consumer(config)

consumer.subscribe(['transactions'])

print("kafka consumer is live and listening for events")

try:
    while True:
        #Poll kafka forr new messages 
        msg = consumer.poll(timeout=1.0)
        
        if msg is None:
            continue
        
        if msg.error():
            if msg.error().code() == KafkaError._PARTITION_EOF:
                continue
            else:
                print(f"Kafka Error encountered: {msg.error()}")
                break
        
        # Deserialize payload and execute the materializtion pipeline     
        try:
            event_data = json.loads(msg.value().decode('utf-8'))
            
            # EXtracct fields 
            user_id = event_data['user_id']
            amount  = float(event_data['amount'])
            timestamp = float(event_data['timestamp'])
            
            materialize_transaction(user_id,amount,timestamp)
            print(f"successfully materialized  transcation for user {user_id}")
            
        except ( json.JSONDecodeError,KeyError,ValueError) as parse_err:
            print(f"skipping malformed message . parse error {parse_err}")
            continue
except KeyboardInterrupt:
    print("Shutting down  consumer cleanly")
finally:
    consumer.close()