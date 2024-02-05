import copy
import numpy as np
import time
import itertools
import math
from typing import List, Optional, Union, Tuple

from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.trajectory import Trajectory
from commonroad.scenario.state import InitialState, State
from commonroad.prediction.prediction import TrajectoryPrediction
from commonroad.scenario.obstacle import DynamicObstacle, ObstacleType, StaticObstacle
from commonroad.geometry.shape import Rectangle, Circle

from planner import Planner
from sensor import Sensor
from occlusion_tracker import Occlusion_tracker

import yaml

from utilities import add_no_stop_zone_DEU_Ffb
from dataclasses import dataclass

from nav_msgs.msg import Odometry


@dataclass
class VehicleDetection:
    position: np.array  # [x, y] m
    orientation: float  # rad
    length: float  # m
    width: float  # m


def euler_from_quaternion(x, y, z, w):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)

    return roll_x, pitch_y, yaw_z  # in radians


def quaternion_from_yaw(yaw):
    qx = 0
    qy = 0
    qz = np.sin(yaw / 2)
    qw = np.cos(yaw / 2)

    return [qx, qy, qz, qw]


def update_ego_vehicle(ego_vehicle: DynamicObstacle, state: State):
    """Update the ego vehicle based on the state and not based on the predicted trajectory."""
    if ego_vehicle.prediction is not None:
        trajectory = Trajectory(1 + state.time_step, ego_vehicle.prediction.trajectory.state_list[1:])
        trajectory_prediction = TrajectoryPrediction(trajectory, ego_vehicle.obstacle_shape)
    else:
        trajectory_prediction = None

    return DynamicObstacle(
        obstacle_id=ego_vehicle.obstacle_id,
        obstacle_type=ego_vehicle.obstacle_type,
        obstacle_shape=ego_vehicle.obstacle_shape,
        initial_state=state,
        prediction=trajectory_prediction,
    )


from commonroad.common.util import Interval, AngleInterval
from commonroad.geometry.shape import Shape


@dataclass
class Detection:
    position: List # [x, y]
    orientation: float
    shape: Optional[Union[List, Tuple]] = None # [length, width]

    def __post_init__(self):
        self.position = np.array(self.position)


def get_detected_obstacles(scenario: Scenario, configuration, time_step, is_sim: bool = True) -> List:
    """Get a list of obstacles and shapes --> see: src > sensors.py > Sensor > get_observed_obstacles()"""
    if not is_sim:
        # TODO: obtain detections from a ROS topic obtain by the DATMO package
        pass
    else:
        detections = [
            Detection([-50, -2], 0),
            Detection([-30, -2], 0),
            Detection([15, -25], 0, [15, 15])
        ]  # Eventually an array from DATMO package
        # TODO: make a dummy implementation
        observed_obstacles = []  # commonroad obstacles

        for detection in detections:
            if detection.shape is None:
                obstacle_shape = Rectangle(
                    configuration.get("obstacle_min_length"), configuration.get("obstacle_min_width")
                )
            else:
                obstacle_shape = Rectangle(*detection.shape)

            position: Shape = Circle(0.1, detection.position)
            orientation: AngleInterval = AngleInterval(
                detection.orientation - configuration.get("orientation_margin"),
                detection.orientation + configuration.get("orientation_margin"),
            )
            velocity_range: Interval = Interval(configuration.get("min_velocity"), configuration.get("max_velocity"))

            obstacle_state = InitialState(
                time_step=time_step,
                # position=position,
                # orientation=orientation,
                # velocity=velocity_range,
                position=detection.position,
                orientation=detection.orientation,
            )
            obstacle = DynamicObstacle(
                obstacle_id=scenario.generate_object_id(),
                obstacle_type=ObstacleType.CAR,
                obstacle_shape=obstacle_shape,
                initial_state=obstacle_state,
            )
            observed_obstacles.append(obstacle)

    return observed_obstacles


def get_ego_vehicle_odometry(ego_vehicle, is_sim: bool = True) -> Odometry:
    """Get the ROS message with the odometry of the ego vehicle. When simulating, use the predicted trajectory."""
    if not is_sim:
        # TODO: subscribe to a ROS topic
        pass
    else:
        # Simulation: Odometry message based on trajectory
        if ego_vehicle.prediction is not None:
            state = ego_vehicle.prediction.trajectory.state_list[0]
        else:
            state = ego_vehicle.initial_state
        return odom_from_state(state)


def odom_from_state(state: InitialState) -> Odometry:
    odom = Odometry()
    odom.pose.pose.position.x = state.position[0]
    odom.pose.pose.position.y = state.position[1]

    # TODO: Orientation as quaternion
    yaw = state.orientation
    qx, qy, qz, qw = quaternion_from_yaw(yaw)
    odom.pose.pose.orientation.x = qx
    odom.pose.pose.orientation.y = qy
    odom.pose.pose.orientation.z = qz
    odom.pose.pose.orientation.w = qw

    velocity = state.velocity * np.array([np.cos(yaw), np.sin(yaw)])
    # velocity = np.clip(velocity, 0, state)
    correction_factor = 0.999999  # FIXME: velocity should never exceed max velocity (not even slightly)
    odom.twist.twist.linear.x = velocity[0] * correction_factor
    odom.twist.twist.linear.y = velocity[1] * correction_factor

    return odom


def state_from_odom(odom: Odometry, time_step: int = 0) -> State:
    """Convert a ROS odometry message to a commonroad state."""
    position = [
        odom.pose.pose.position.x,
        odom.pose.pose.position.y,
    ]
    quaternion = [
        odom.pose.pose.orientation.x,
        odom.pose.pose.orientation.y,
        odom.pose.pose.orientation.z,
        odom.pose.pose.orientation.w,
    ]
    _, _, yaw = euler_from_quaternion(*quaternion)
    velocity = [
        odom.twist.twist.linear.x,
        odom.twist.twist.linear.y,
    ]
    velocity_along_heading = np.array([np.cos(yaw), np.sin(yaw)]) @ velocity
    return InitialState(
        position=np.array(position),
        orientation=yaw,  # along x is 0, ccw is positive
        velocity=velocity_along_heading,
        time_step=time_step,
    )


def step_simulation(scenario, configuration):
    start_time = time.time()  # for debugging
    driven_state_list = []
    percieved_scenarios = []
    sensor_views = []

    ego_vehicle_initial_state = InitialState(
        position=np.array([configuration.get("initial_state_x"), configuration.get("initial_state_y")]),
        orientation=configuration.get("initial_state_orientation"),
        velocity=configuration.get("initial_state_velocity"),
        time_step=0,
    )
    ego_shape = Rectangle(configuration.get("vehicle_length"), configuration.get("vehicle_width"))
    # ego_vehicle_initial_state = InitialState(position=np.array([0, 0]), orientation=0, velocity=0, time_step=0)
    ego_vehicle = DynamicObstacle(scenario.generate_object_id(), ObstacleType.CAR, ego_shape, ego_vehicle_initial_state)

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
    # car_shape = Rectangle(configuration.get("vehicle_length"), configuration.get("vehicle_width"))
    # obstacle_state1 = InitialState(position=np.array([7, -18]), orientation=0, time_step=0)
    # parked_car1 = StaticObstacle(
    #     scenario.generate_object_id(),
    #     ObstacleType.CAR,
    #     car_shape,
    #     obstacle_state1,
    # )
    # obstacle_state2 = InitialState(position=np.array([-6, -18]), orientation=np.pi / 2, time_step=0)
    # parked_car2 = StaticObstacle(
    #     scenario.generate_object_id(),
    #     ObstacleType.CAR,
    #     car_shape,
    #     obstacle_state2,
    # )
    # scenario.add_objects([parked_car1, parked_car2])
    # # scenario.add_objects(parked_car2)
    # # scenario.remove_obstacle(parked_car2)

    # ### Add my own dynamic obstacle
    # driving_car_initial_state = InitialState(position=np.array([20, -2]), orientation=0, time_step=0)
    # # trajectory = Trajectory(0, [obstacle_initial_state, obstacle_initial_state])
    # # driving_car_pred = TrajectoryPrediction(trajectory, car_shape)
    # driving_car = DynamicObstacle(
    #     obstacle_id=scenario.generate_object_id(),
    #     obstacle_type=ObstacleType.CAR,
    #     obstacle_shape=car_shape,
    #     initial_state=driving_car_initial_state,
    # )
    # scenario.add_objects(driving_car)

    for time_step in itertools.count():
        # TODO: keep track of the time_step -- verify that it works
        # TODO: obtain the detected obstacles from a ROS topic
        percieved_scenario = copy.deepcopy(scenario)
        detected_obstacles = get_detected_obstacles(percieved_scenario, configuration, time_step)

        # TODO: obtain the ego vehicle state from a ROS topic
        ego_vehicle_odom = get_ego_vehicle_odometry(ego_vehicle)
        ego_vehicle_state = state_from_odom(ego_vehicle_odom, time_step)
        ego_vehicle = update_ego_vehicle(ego_vehicle, ego_vehicle_state)  # only update for visualization ... ??!!

        # Start with an empty percieved scenario
        # for obstacle in percieved_scenario.obstacles:  # TODO: ensure scenario is always empty
        #     percieved_scenario.remove_obstacle(obstacle)

        percieved_scenario.add_objects(detected_obstacles)

        # Update the sensor and get the sensor view and the list of observed obstacles
        sensor.update(ego_vehicle.initial_state)
        sensor_view = sensor.get_sensor_view(percieved_scenario)

        # Update the tracker with the new sensor view and get the prediction for the shadows
        occ_track.update(sensor_view, ego_vehicle.initial_state.time_step)
        shadow_obstacles = occ_track.get_dynamic_obstacles(percieved_scenario)
        percieved_scenario.add_objects(shadow_obstacles)

        # Update the planner and plan a trajectory
        add_no_stop_zone_DEU_Ffb(
            percieved_scenario, time_step + configuration.get("planning_horizon"), configuration.get("safety_margin")
        )  # should not be necessary in every timestep
        planner.update(ego_vehicle.initial_state)
        collision_free_trajectory = planner.plan(percieved_scenario)
        if collision_free_trajectory:
            ego_vehicle.prediction = collision_free_trajectory
        # else, if no trajectory found, keep previous collision free trajectory
        # TODO: send motor commands based on trajectory

        # TODO: save to disk every timestep to be able to kill the process at any time
        percieved_scenario.add_objects(ego_vehicle)
        percieved_scenarios.append(percieved_scenario)
        sensor_views.append(sensor_view)
        driven_state_list.append(ego_vehicle.initial_state)

        # TODO: RVIZ visualization

        if time_step > simulation_steps:
            break

    print(f"Simulation took: {time.time() - start_time} s")

    # Set initial_state to initial state and not current
    ego_vehicle.initial_state = driven_state_list.pop(0)
    driven_trajectory = Trajectory(1, driven_state_list)
    driven_trajectory_pred = TrajectoryPrediction(driven_trajectory, ego_vehicle.obstacle_shape)
    ego_vehicle.prediction = driven_trajectory_pred
    return ego_vehicle, percieved_scenarios, sensor_views
