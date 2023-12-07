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
    plt.xlim(-60,60)
    plt.ylim(-50,10)

with open("my_scenario/config_MyScenario.yaml") as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
    
scenario1, _ = CommonRoadFileReader("my_scenario/ZAM_MyIntersection-1_1_T-1.xml").open()
scenario2, _ = CommonRoadFileReader("my_scenario/ZAM_MyIntersection-1_1_T-1.xml").open()

track_vehicle, tracked_scenarios, tracked_views = step_simulation(scenario1, config)
config['tracking_enabled'] = False
no_track_vehicle, not_tracked_scenarios, not_tracked_views = step_simulation(scenario2, config)

t1 = 0
t2 = 3
fig, ax = plt.subplots(2, 2, figsize=(20,13))
plt.sca(ax[0][0])
plot(t1, no_track_vehicle, not_tracked_scenarios, not_tracked_views)
plt.sca(ax[0][1])
plot(t2, no_track_vehicle, not_tracked_scenarios, not_tracked_views)
plt.sca(ax[1][0])
plot(t1, track_vehicle, tracked_scenarios, tracked_views)
plt.sca(ax[1][1])
plot(t2, track_vehicle, tracked_scenarios, tracked_views)