import json

def read_json():
    with open('task.json', 'r') as f:
        return json.load(f)

if __name__ == "__main__":
    print(read_json())
