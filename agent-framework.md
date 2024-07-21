## Scripting and AI
- **Python Integration**: Expose a rich API to the Python interpreter, allowing for flexible scripting and AI control.
- **AI Bots**: Implement a base class `Agent` and derived class `SubsumptionAgent` to facilitate the creation of reactive, intelligent bots.


##### AI Bot Framework
- Define a base class `Agent` and a derived class `SubsumptionAgent` for creating intelligent, reactive bots.
- Enable flexible AI scripting through Python, allowing for rapid development and testing of AI behaviors.

```python
class Agent:
    def __init__(self, id):
        self.id = id

    def update(self):
        pass

class SubsumptionAgent(Agent):
    def __init__(self, id):
        super().__init__(id)
        self.behaviors = []

    def add_behavior(self, behavior):
        self.behaviors.append(behavior)

    def update(self):
        for behavior in self.behaviors:
            if behavior.should_run():
                behavior.run()
                break
```
