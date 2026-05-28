from datetime import datetime
from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from airflow.providers.standard.operators.bash import BashOperator
from airflow.providers.common.sql.operators.sql import SQLExecuteQueryOperator


def _materalize_feature():
            print("Extracting , aggregating , and writing fresh features to redis ... ")
            
            
with DAG(
    dag_id='feature_store_nightly_materialization',
    schedule='@daily',
    start_date=datetime(2024, 1, 1),
    catchup=False
) as dag:
    

        check_new_data = SQLExecuteQueryOperator(
            task_id='check_new_data',
            conn_id='postgres_default',
            sql="""
            SELECT CASE
                WHEN COUNT(*) > 0 THEN 1
                ELSE 1/0
            END
            FROM feature_store
            WHERE computed_at >= NOW() - INTERVAL '1 DAY';
"""
        )
        
        
        materalize_feature = PythonOperator(
            task_id='materialize_features',
            python_callable = _materalize_feature
        )
        
        #Log 
        log_completion = BashOperator(
            task_id='log_completion',
            bash_command='echo "Feature  store materialization successfully at $(date)."'
        )
        
        check_new_data >> materalize_feature >> log_completion