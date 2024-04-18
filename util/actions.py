import csv
import json
import json
import os

class Action:
    def __init__(self) -> None:
        self.action = []
        try:
            with open("data/actions_dataset/actions.json", 'r') as f:
                data = json.load(f)
            for action in data['all_actions']:
                self.action.append(action)

            pb_file = "data/actions_dataset/activity_net.pbtxt"
            
            if os.path.exists(pb_file):
                os.remove(pb_file)
            
            pb_text = open(pb_file, 'a', newline='')
            for i in range(0, len(self.action)):
                pb_text.write("item {\n")
                pb_text.write(f"  name: \"{self.action[i]}\"\n")
                pb_text.write(f"  id: {i}\n")
                pb_text.write("}\n")
            pb_text.close()

        except FileNotFoundError:
            print("File not found")

    
    def append_to_actions_list(self, name) :
        if (name not in self.action):
            self.action.append(name)
        with open("data/actions_dataset/actions.json", 'w') as f:
            json.dump({"all_actions": self.action}, f)


Action()