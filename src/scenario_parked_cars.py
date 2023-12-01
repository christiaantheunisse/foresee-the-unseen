import sys  
sys.path.insert(0, 'src/')

from commonroad.common.file_reader import CommonRoadFileReader
from simulate_trajectory import step_simulation
from visualizer import Visualizer
import yaml
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML

def plot(time_step, ego_vehicle, scenarios, sensor_views):
    plt.cla()
    Visualizer().plot(scenario=scenarios[time_step],
                  sensor_view=sensor_views[time_step],
                  ego_vehicle=scenarios[time_step].obstacle_by_id(ego_vehicle.obstacle_id),
                  time_begin=time_step)
    plt.axis('scaled')
    plt.xlim(0,120)
    plt.ylim(-40,40)


if __name__ == "__main__":
    # with open("scenario_parked_cars/config_parked.yaml") as file:
    with open("my_scenario/config_MyScenario.yaml") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    # scenario1, _ = CommonRoadFileReader("scenario_parked_cars/DEU_Ffb-1_parked.xml").open()
    # scenario2, _ = CommonRoadFileReader("scenario_parked_cars/DEU_Ffb-1_parked.xml").open()
    scenario1, _ = CommonRoadFileReader("my_scenario/ZAM_MyIntersection-1_1_T-1.xml").open()
    scenario2, _ = CommonRoadFileReader("my_scenario/ZAM_MyIntersection-1_1_T-1.xml").open()
    track_vehicle, tracked_scenarios, tracked_views = step_simulation(scenario1, config)