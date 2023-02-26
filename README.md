I use python 3.8.8
follow this steps:
1. install requirements.txt
2. run scrap.py
3. run find_topic.py
4. label topic manually (title_topic_label.csv)
5. run find_label.py
6. run train_severity_model.py
7. run train_stack_group_model.py
8. run api `uvicorn api:app --reload`
9. open `http://localhost:8000/docs`
