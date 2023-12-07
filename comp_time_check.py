import sys

sys.path.insert(0, "src/")

from commonroad.common.file_reader import CommonRoadFileReader
from simulate_trajectory import step_simulation
import yaml

with open("my_scenario/config_MyScenario.yaml") as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

scenario1, _ = CommonRoadFileReader("my_scenario/ZAM_MyIntersection-1_1_T-1.xml").open()
scenario2, _ = CommonRoadFileReader("my_scenario/ZAM_MyIntersection-1_1_T-1.xml").open()


def print_time_dict(time_dict):
    for key, (steps, values) in time_dict.items():
        print("-" * 80)
        print(key)
        print("-" * 80)
        for step, value in zip(steps, values):
            print("\t" + step + " " * (30 - len(step)) + str(round(value * 1000, 2)) + 'ms')
        print()

def mean_time_dicts2(time_dicts):
    avg_dict = {}
    for time_dict in time_dicts:
        for key, (steps, values) in time_dict.items():
            try:
                avg_dict[key] = (steps, avg_dict[key][1] + values)
            except KeyError:
                avg_dict[key] = (steps, values)

    for key, (steps, values) in avg_dict.items():
        avg_dict[key] = (steps, values / len(time_dicts))

    return avg_dict

def mean_time_dicts(time_dicts):
    avg_dict = {}
    for time_dict in time_dicts:
        for key, (_, values) in time_dict.items():
            try:
                avg_dict[key] = avg_dict[key] + values
            except KeyError:
                avg_dict[key] = values

    # take average
    for key, values in avg_dict.items():
        avg_dict[key] = values / len(time_dicts)

    # add names
    for key, (steps, _) in time_dicts[0].items():
        avg_dict[key] = (steps, avg_dict[key])

    return avg_dict

time_dicts = []
for _ in range(10):
    _, _, _, time_dict = step_simulation(scenario1, config)
    time_dicts.append(time_dict)

avg_time_dict = mean_time_dicts(time_dicts)
print_time_dict(avg_time_dict)
