import json

def get_task():
    with open('task.json', 'r') as f:
        return json.load(f)

if __name__ == "__main__":
    task = get_task()
    print("Axes constraints:", task.get('constraints'))
