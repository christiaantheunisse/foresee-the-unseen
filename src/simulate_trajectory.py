import copy
import numpy as np
import time
from typing import List
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.scenario.trajectory import Trajectory
from commonroad.scenario.state import InitialState
from commonroad.prediction.prediction import TrajectoryPrediction
from commonroad.scenario.obstacle import DynamicObstacle, ObstacleType, StaticObstacle
from commonroad.geometry.shape import Rectangle

from planner import Planner
from sensor import Sensor
from occlusion_tracker import Occlusion_tracker

import yaml

from utilities import add_no_stop_zone_DEU_Ffb
from dataclasses import dataclass


@dataclass
class VehicleDetection:
    position: np.array  # [x, y] m
    orientation: float  # rad
    length: float  # m
    width: float  # m


# Create new scenario with new vehicles at next time step
def step_scenario(scenario):
    new_scenario = copy.deepcopy(scenario)

    ### Update vehicles based on sensor measurements
    # Remove the old detections
    # for vehicle in scenario.dynamic_obstacles:
    #     new_scenario.remove_obstacle(vehicle)

    # print(f"TIME STEP: {current_time_step}")
    # # Add the newly detected obstacles
    # for vehicle in detected_vehicles:
    #     vehicle_cm = DynamicObstacle(obstacle_id=scenario.generate_object_id(),
    #                                 obstacle_type=ObstacleType.CAR,
    #                                 obstacle_shape=Rectangle(vehicle.length, vehicle.width),
    #                                 initial_state=InitialState(position=vehicle.position,
    #                                                         orientation=vehicle.orientation,
    #                                                         time_step=current_time_step + 1))
    #     new_scenario.add_objects(vehicle_cm)

    # print(f"No. of vehicles: {len(scenario.dynamic_obstacles)}")
    for vehicle in scenario.dynamic_obstacles:
        new_scenario.remove_obstacle(vehicle)
        new_position = vehicle.initial_state.position + np.array([-1, 0])
        new_state = InitialState(
            time_step=vehicle.initial_state.time_step + 1,
            position=new_position,
            orientation=vehicle.initial_state.orientation,
        )
        stepped_vehicle = DynamicObstacle(
            obstacle_id=vehicle.obstacle_id,
            obstacle_type=vehicle.obstacle_type,
            obstacle_shape=vehicle.obstacle_shape,
            initial_state=new_state,
        )
        new_scenario.add_objects(stepped_vehicle)

    # for vehicle in scenario.dynamic_obstacles:
    #     new_scenario.remove_obstacle(vehicle)
    #     if len(vehicle.prediction.trajectory.state_list) > 1:
    #         stepped_vehicle = step_vehicle(vehicle)
    #         new_scenario.add_objects(stepped_vehicle)
    return new_scenario


def step_vehicle(vehicle):
    initial_state = vehicle.prediction.trajectory.state_list[0]
    trajectory = Trajectory(1 + initial_state.time_step, vehicle.prediction.trajectory.state_list[1:])
    return DynamicObstacle(
        vehicle.obstacle_id,
        vehicle.obstacle_type,
        vehicle.obstacle_shape,
        initial_state,
        TrajectoryPrediction(trajectory, vehicle.obstacle_shape),
    )


def step_simulation(scenario, configuration):
    driven_state_list = []
    percieved_scenarios = []
    sensor_views = []

    ego_shape = Rectangle(configuration.get("vehicle_length"), configuration.get("vehicle_width"))
    ego_initial_state = InitialState(
        position=np.array([configuration.get("initial_state_x"), configuration.get("initial_state_y")]),
        orientation=configuration.get("initial_state_orientation"),
        velocity=configuration.get("initial_state_velocity"),
        time_step=0,
    )
    ego_vehicle = DynamicObstacle(scenario.generate_object_id(), ObstacleType.CAR, ego_shape, ego_initial_state)

    sensor = Sensor(
        ego_vehicle.initial_state.position,
        field_of_view=configuration.get("field_of_view_degrees") * 2 * np.pi / 360,
        min_resolution=configuration.get("min_resolution"),
        view_range=configuration.get("view_range"),
    )

    occ_track = Occlusion_tracker(
        scenario,
        min_vel=configuration.get("min_velocity"),
        max_vel=configuration.get("max_velocity"),
        min_shadow_area=configuration.get("min_shadow_area"),
        prediction_horizon=configuration.get("prediction_horizon"),
        tracking_enabled=configuration.get("tracking_enabled"),
    )

    planner = Planner(
        ego_vehicle.initial_state,
        vehicle_shape=ego_vehicle.obstacle_shape,
        goal_point=[configuration.get("goal_point_x"), configuration.get("goal_point_y")],
        reference_speed=configuration.get("reference_speed"),
        max_acceleration=configuration.get("max_acceleration"),
        max_deceleration=configuration.get("max_deceleration"),
        time_horizon=configuration.get("planning_horizon"),
    )
    simulation_steps = configuration.get("simulation_duration")

    ### Add my own static obstacle
    """
        A static obstacle occludes the road there were the obstacle is placed, so no need for additional modelling.
        This does not working with plotting, so change to a dynamic obstacle.
    """
    car_shape = Rectangle(configuration.get("vehicle_length"), configuration.get("vehicle_width"))
    obstacle_state1 = InitialState(position=np.array([7, -18]), orientation=0, time_step=0)
    parked_car1 = StaticObstacle(
        scenario.generate_object_id(),
        ObstacleType.CAR,
        car_shape,
        obstacle_state1,
    )
    obstacle_state2 = InitialState(position=np.array([-6, -18]), orientation=np.pi / 2, time_step=0)
    parked_car2 = StaticObstacle(
        scenario.generate_object_id(),
        ObstacleType.CAR,
        car_shape,
        obstacle_state2,
    )
    scenario.add_objects([parked_car1, parked_car2])
    # scenario.add_objects(parked_car2)
    # scenario.remove_obstacle(parked_car2)

    ### Add my own dynamic obstacle
    driving_car_initial_state = InitialState(position=np.array([20, -2]), orientation=0, time_step=0)
    # trajectory = Trajectory(0, [obstacle_initial_state, obstacle_initial_state])
    # driving_car_pred = TrajectoryPrediction(trajectory, car_shape)
    driving_car = DynamicObstacle(
        obstacle_id=scenario.generate_object_id(),
        obstacle_type=ObstacleType.CAR,
        obstacle_shape=car_shape,
        initial_state=driving_car_initial_state,
    )
    scenario.add_objects(driving_car)
    # obstacle_prediction = SetBasedPrediction(self.time_step+1, occupancy_set)

    init_time = time.time()
    time_steps = []
    time_steps_shadows, no_shadows = [], []
    time_occupancy_sets = []
    for step in range(simulation_steps + 1):
        t_steps = [time.time()]  # log runtime

        # Start with an empty percieved scenario
        percieved_scenario = copy.deepcopy(scenario)
        for obstacle in percieved_scenario.obstacles:
            percieved_scenario.remove_obstacle(obstacle)
        t_steps.append(time.time())  # log runtime

        # Update the sensor and get the sensor view and the list of observed obstacles
        sensor.update(ego_vehicle.initial_state)  # initial_state is current state
        sensor_view = sensor.get_sensor_view(scenario)
        observed_obstacles, _ = sensor.get_observed_obstacles(sensor_view, scenario)
        percieved_scenario.add_objects(observed_obstacles)

        t_steps.append(time.time())  # log runtime

        # Update the tracker with the new sensor view and get the prediction for the shadows
        ### THIS code uses at least 95% of the time
        ## 5% of the 95%
        occ_track.update(sensor_view, ego_vehicle.initial_state.time_step)
        ## 95% of the 95%
        shadow_obstacles, t_steps_shadows, t_occupancy_sets = occ_track.get_dynamic_obstacles(percieved_scenario)
        no_shadows.append(t_steps_shadows.shape[0])
        time_steps_shadows.append(t_steps_shadows.mean(axis=0))
        time_occupancy_sets.append(t_occupancy_sets.mean())
        ## 0% of the 95%
        percieved_scenario.add_objects(shadow_obstacles)
        t_steps.append(time.time())  # log runtime

        # Update the planner and plan a trajectory
        # if
        add_no_stop_zone_DEU_Ffb(
            percieved_scenario, step + configuration.get("planning_horizon"), configuration.get("safety_margin")
        )
        planner.update(ego_vehicle.initial_state)
        collision_free_trajectory = planner.plan(percieved_scenario)
        if collision_free_trajectory:
            ego_vehicle.prediction = collision_free_trajectory
        # else, if no trajectory found, keep previous collision free trajectory
        t_steps.append(time.time())  # log runtime

        # Add the ego vehicle to the perceived scenario
        percieved_scenario.add_objects(ego_vehicle)

        percieved_scenarios.append(percieved_scenario)
        sensor_views.append(sensor_view)
        driven_state_list.append(ego_vehicle.initial_state)

        ego_vehicle = step_vehicle(ego_vehicle)

        scenario = step_scenario(scenario)
        t_steps.append(time.time())  # log runtime

        # print(f"Step {step}: took {time.time() - t_steps[0]} s")
        t_steps = np.array(t_steps)
        t_steps = t_steps[1:] - t_steps[:-1]
        time_steps.append(t_steps)

    # print(f"Simulation took: {time.time() - init_time} s")
    time_steps = np.array(time_steps)
    time_steps_shadows, no_shadows = np.array(time_steps_shadows), np.array(no_shadows)
    time_occupancy_sets = np.array(time_occupancy_sets)
    time_dict = {
        f"Time for steps main simulation loop [{time_steps.shape[0]} times]": (
            ["copy scenario", "read sensor", "predict shadows", "plan traject", "save scenario"],
            time_steps.mean(axis=0),
        ),
        f"Time for `predict shadows` [{no_shadows.mean()} times per `main simulation`]": (
            ["get_occupanc_set", "..."],
            time_steps_shadows.mean(axis=0),
        ),
        f"Time for `get_occupanc_set` [1 time per `shadow predict`]": (
            ["for loop occlusion prediction"],
            time_occupancy_sets.mean(axis=0).reshape(1),
        )
    }

    # Set initial_state to initial state and not current
    ego_vehicle.initial_state = driven_state_list.pop(0)
    driven_trajectory = Trajectory(1, driven_state_list)
    driven_trajectory_pred = TrajectoryPrediction(driven_trajectory, ego_vehicle.obstacle_shape)
    ego_vehicle.prediction = driven_trajectory_pred
    return ego_vehicle, percieved_scenarios, sensor_views, time_dict
