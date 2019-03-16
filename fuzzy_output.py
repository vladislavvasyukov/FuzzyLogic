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

decision = control.Consequent(arange(0, 20, 1), 'decision')
decision.automf(number=3, names=['low', 'middle', 'high'])

prereq_one = (
    state_health['bad']
    & (calories['little'] | calories['enough'])
    & (training_experience['small'] | training_experience['medium'])
)
rule_one = control.Rule(prereq_one, decision['low'])

prereq_two = (
    state_health['good']
    & (
        (calories['little'] & training_experience['small']) 
        | (calories['enough'] & training_experience['small']) 
        | (calories['little'] & training_experience['medium'])
    )
)
rule_two = control.Rule(prereq_two, decision['low'])

prereq_three = (
    (state_health['bad'] & calories['little'] & training_experience['large'])
    | (state_health['bad'] & calories['lot'] & training_experience['small'])
    | (state_health['excellent'] & calories['little'] & training_experience['small'])
)
rule_three = control.Rule(prereq_three, decision['low'])

prereq_four = (
    state_health['excellent']
    & (calories['enough'] | calories['lot'])
    & (training_experience['medium'] | training_experience['large'])
)
rule_four = control.Rule(prereq_four, decision['high'])

prereq_five = (
    training_experience['large']
    & calories['lot']
    & (state_health['bad'] | state_health['good'])
)
rule_five = control.Rule(prereq_five, decision['high'])

prereq_six = (
    (state_health['bad'] & calories['enough'] & training_experience['large'])
    | (state_health['bad'] & calories['lot'] & training_experience['medium'])
    | (state_health['good'] & calories['little'] & training_experience['large'])
    | (state_health['good'] & calories['enough'] & training_experience['medium'])
    | (state_health['good'] & calories['lot'] & training_experience['small'])
    | (state_health['excellent'] & calories['little'] & training_experience['medium'])
    | (state_health['excellent'] & calories['enough'] & training_experience['small'])
)
rule_six = control.Rule(prereq_six, decision['middle'])


prereq_seven = (
    (state_health['excellent'] & calories['little'] & training_experience['large'])
    | (state_health['good'] & calories['enough'] & training_experience['large'])
    | (state_health['good'] & calories['lot'] & training_experience['medium'])
    | (state_health['excellent'] & calories['lot'] & training_experience['small'])
)
rule_seven = control.Rule(prereq_seven, decision['high'])

decision_system = control.ControlSystem([rule_one, rule_two, rule_three, rule_four, rule_five, rule_six, rule_seven])
consultant = control.ControlSystemSimulation(decision_system)

TestInput = namedtuple('TestInput', ['state_health', 'calories', 'training_experience'])

tests = [
    TestInput(5, 800, 19),
    TestInput(5, 800, 1),
    TestInput(15, 1500, 5),
    TestInput(25, 2500, 15),
    TestInput(35, 900, 2),
    TestInput(45, 1600, 6),
    TestInput(55, 2600, 16),
    TestInput(65, 1000, 3),
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

    result = consultant.output['decision']
    print(consultant)
    print("test: {}\nresult: {} hours".format(test, result))
    decision.view(sim=consultant)
    input()
    consultant.reset()
