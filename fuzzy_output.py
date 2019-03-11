from collections import namedtuple
from numpy import arange
from skfuzzy import control, trapmf, smf, zmf


state_health = control.Antecedent(arange(0, 100, 1), 'state_health')
state_health.automf(number=3, names=['bad', 'good', 'excellent'])

calories = control.Antecedent(arange(0, 5000, 1), 'calories')
universe = calories.universe
calories['little'] = zmf(universe, 1000, 1300)
calories['enough'] = trapmf(universe, [1000, 1100, 1800, 2100])
calories['lot'] = smf(universe, 2000, 2300)

training_experience = control.Antecedent(arange(0, 20, 1), 'training_experience')
universe = training_experience.universe
training_experience['small'] = zmf(universe, 1, 4)
training_experience['medium'] = trapmf(universe, [3, 4, 8, 12])
training_experience['large'] = smf(universe, 10, 15)

decision = control.Consequent(arange(0, 100, 1), 'decision')
decision.automf(number=3, names=['low', 'middle', 'high'])

prereq_one = (
    (state_health['good'] | state_health['excellent']) 
    & calories['lot'] 
    & training_experience['large']
)
rule_one = control.Rule(prereq_one, decision['high'])

prereq_two = (
    (state_health['bad'] | state_health['good']) 
    & calories['lot'] 
    & training_experience['small']
) 
rule_two = control.Rule(prereq_two, decision['low'])

prereq_three = (
    state_health['excellent']
    & calories['enough'] 
    & training_experience['medium']
)
rule_three = control.Rule(prereq_three, decision['middle'])

prereq_four = (
    state_health['bad']
    & calories['little'] 
    & training_experience['large']
)
rule_four = control.Rule(prereq_four, decision['middle'])

prereq_five = (
    state_health['excellent']
    & (calories['lot'] | calories['enough']) 
    & training_experience['large']
) 
rule_five = control.Rule(prereq_five, decision['high'])

prereq_six = (
    state_health['bad']
    & calories['little'] 
    & training_experience['small']
)
rule_five = control.Rule(prereq_five, decision['low'])

decision_system = control.ControlSystem([rule_one, rule_two, rule_three, rule_four])
consultant = control.ControlSystemSimulation(decision_system)

TestInput = namedtuple('TestInput', ['state_health', 'calories', 'training_experience'])


tests = [
    TestInput(5, 800, 19),
    # TestInput(5, 800, 1),
    # TestInput(15, 1500, 5),
    TestInput(25, 2500, 15),
    # TestInput(35, 900, 2),
    # TestInput(45, 1600, 6),
    TestInput(55, 2600, 16),
    # TestInput(65, 1000, 3),
    TestInput(75, 1750, 8),
    TestInput(90, 3000, 19),
]

for test in tests:
    print('*'*100)
    consultant.inputs({
        'state_health': test.state_health, 
        'calories': test.calories, 
        'training_experience': test.training_experience
    })
    consultant.compute()

    result = int(round(consultant.output['decision'], 0))
    print("test: {}\nresult: {}%".format(test, result))
    decision.view(sim=consultant)
    input()
    consultant.reset()
