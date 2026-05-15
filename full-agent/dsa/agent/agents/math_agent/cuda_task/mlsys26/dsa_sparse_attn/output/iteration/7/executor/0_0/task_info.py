import json

def get_task():
    with open('task.json', 'r') as f:
        return json.load(f)

if __name__ == "__main__":
    task = get_task()
    print("Task Name:", task.get('name'))
    print("Axes:", task.get('axes'))
    print("Inputs:", list(task.get('inputs', {}).keys()))
    print("Outputs:", list(task.get('outputs', {}).keys()))
