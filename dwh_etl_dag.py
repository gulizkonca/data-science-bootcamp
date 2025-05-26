Python 3.13.2 (v3.13.2:4f8bb3947cf, Feb  4 2025, 11:51:10) [Clang 15.0.0 (clang-1500.3.9.4)] on darwin
Type "help", "copyright", "credits" or "license()" for more information.
>>> # airflow dags/dwh_etl_dag.py
... # pip install pandas sqlalchemy psycopg2-binary requests
... from airflow import DAG
... from airflow.operators.python import PythonOperator
... from airflow.utils.dates import days_ago
... import pandas as pd
... from sqlalchemy import create_engine
... import requests, io, datetime as dt
... 
... URL = "https://example.com/daily_data.json"
... PG_CONN = "postgresql+psycopg2://user:pass@postgres:5432/dwh"
... 
... def extract(**context):
...     r = requests.get(URL, timeout=30)
...     r.raise_for_status()
...     context["ti"].xcom_push(key="json", value=r.text)
... 
... def transform(**context):
...     raw = context["ti"].xcom_pull(key="json")
...     df = pd.read_json(io.StringIO(raw))
...     df["ingest_ts"] = dt.datetime.utcnow()
...     context["ti"].xcom_push(key="df", value=df.to_json())
... 
... def load(**context):
...     df = pd.read_json(io.StringIO(context["ti"].xcom_pull(key="df")))
...     engine = create_engine(PG_CONN)
...     df.to_sql("raw_events", engine, if_exists="append", index=False)
... 
... with DAG(
...     dag_id="dwh_etl_json_to_pg",
...     start_date=days_ago(1),
...     schedule_interval="0 2 * * *",
...     catchup=False,
... ) as dag:
...     t1 = PythonOperator(task_id="extract", python_callable=extract)
...     t2 = PythonOperator(task_id="transform", python_callable=transform)
...     t3 = PythonOperator(task_id="load", python_callable=load)
...     t1 >> t2 >> t3
