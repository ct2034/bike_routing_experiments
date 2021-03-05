#!/usr/bin/env python3
import math

import gpxpy
import numpy as np
import routingpy
from geopy import Point
from geopy.distance import geodesic
from matplotlib import pyplot as plt

# Idea:
#   From a central point, sample points within a circle and plan a route along
#   them. Evaluate roundness and change points until good.
GENERATIONS = 20


def show_path(path, points):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(path[:, 0], path[:, 1], 'k')
    plt.plot(points[:, 0], points[:, 1], 'r.')
    ax.set_aspect('equal')
    plt.show()


def write_gpx(fname: str, points: np.ndarray):
    gpx = gpxpy.gpx.GPX()
    gpx_track = gpxpy.gpx.GPXTrack()
    gpx.tracks.append(gpx_track)
    gpx_segment = gpxpy.gpx.GPXTrackSegment()
    gpx_track.segments.append(gpx_segment)
    for p in points:
        gpx_segment.points.append(
            gpxpy.gpx.GPXTrackPoint(p[1], p[0]))
    with open(fname, 'w') as f:
        f.write(gpx.to_xml())


def get_len_in_km(dir) -> float:
    return dir.distance / 1000


def measure_roundness_error(center: np.ndarray, radius: float,
                            points: np.ndarray) -> float:
    # return a number of how well the `points` lie on a circle of `radius`
    # around `center`. The higher, the worse
    error: float = 0
    for point in points:
        dist = geodesic(
            (center[1], center[0]),
            (point[1], point[0])
        ).km
        error += math.pow(dist - radius, 2)
    return error


def get_points_from_radiants(center: np.ndarray, radius: float,
                             radiants: np.ndarray) -> np.ndarray:
    # using `radiants`, project point in `radius` around `center`
    # adding beginning at the end for full circle
    radiants_with_end = np.append(radiants, radiants[0])
    points = np.array(list(map(
        lambda x: list(
            geodesic(kilometers=radius).destination(
                Point(center[1], center[0]), math.degrees(x))
        ),
        radiants_with_end)))
    return points[:, [1, 0]]


def mutate_radiants(noise: float, radiants: np.ndarray) -> np.ndarray:
    rnd = np.random.normal(0, scale=noise, size=radiants.shape)
    keep = np.random.random(size=radiants.shape)
    rnd[np.where(keep < .8)] = 0
    return radiants + rnd


def mutate_points(noise: float, points: np.ndarray) -> np.ndarray:
    rnd = np.random.normal(0, scale=noise, size=points.shape)
    keep = np.random.random(size=points.shape[0])
    rnd[np.where(keep < .8)] = 0
    return points + rnd


def get_path_from_points(gh, points: np.ndarray) -> np.ndarray:
    dir = gh.directions(points, profile='bike')
    path = np.array(dir.geometry)
    return path


def get_path_from_radiants(gh, center: np.ndarray, radius: float,
                           radiants: np.ndarray) -> np.ndarray:
    points = get_points_from_radiants(center, radius, radiants)
    path = get_path_from_points(gh, points)
    return path


def eval_points(gh, center: np.ndarray, radius: float,
                points: np.ndarray) -> float:
    dir = gh.directions(points, profile='bike')
    cost = get_len_in_km(dir)  # shortening it now
    return cost


def eval_radiants(gh, center: np.ndarray, radius: float,
                  radiants: np.ndarray) -> float:
    path = get_path_from_radiants(gh, center, radius, radiants)
    cost = measure_roundness_error(center, radius, path)
    return cost


def optimize_radiants(gh, center, radius, best_radiants):
    noise = math.pi / len(best_radiants)
    n_generations = GENERATIONS
    n_samples = 20
    best_cost = eval_radiants(gh, center, radius, best_radiants)
    print(f'initial_cost: {best_cost}')
    for gen in range(n_generations):
        print(f'generation {gen}')
        print(f'noise: {noise}')
        costs = []
        radiantss = []
        for smpl in range(n_samples):
            radiants = mutate_radiants(noise, best_radiants)
            cost = eval_radiants(gh, center, radius, radiants)
            radiantss.append(radiants)
            costs.append(cost)
        min_i = np.argmin(cost)
        if costs[min_i] < best_cost:
            print(f'New best cost! old: {best_cost}, new: {costs[min_i]}')
            best_cost = costs[min_i]
            best_radiants = radiantss[min_i]
        else:
            print(f'no new best cost. old: {best_cost}, new: {costs[min_i]}')
        noise = .95 * noise
    return best_radiants


def optimize_points(gh, center, radius, best_points):
    noise = .001 * radius * 2 * math.pi / len(best_points)
    n_generations = 2*GENERATIONS
    n_samples = 30
    best_cost = eval_points(gh, center, radius, best_points)
    print(f'initial_cost: {best_cost}')
    for gen in range(n_generations):
        print(f'generation {gen}')
        print(f'noise: {noise}')
        costs = []
        pointss = []
        for smpl in range(n_samples):
            points = mutate_points(noise, best_points)
            cost = eval_points(gh, center, radius, points)
            pointss.append(points)
            costs.append(cost)
        min_i = np.argmin(cost)
        if costs[min_i] < best_cost:
            print(f'New best cost! old: {best_cost}, new: {costs[min_i]}')
            best_cost = costs[min_i]
            best_points = pointss[min_i]
        else:
            print(f'no new best cost. old: {best_cost}, new: {costs[min_i]}')
        noise = .95 * noise
    return best_points


if __name__ == "__main__":
    gh = routingpy.Graphhopper(base_url="http://localhost:8989")
    center = [9.30619239807129, 48.74161597751605]  # ES
    radius = 10  # km
    n_points = 30

    # evenly distributed angles around the circle
    radiants = np.linspace(0, 2*math.pi, n_points-1, endpoint=False)

    # firstly optimize the angle to get a route as circular as possible
    best_radiants = optimize_radiants(
        gh, center, radius, radiants)
    points = get_points_from_radiants(center, radius, best_radiants)
    # then optimize the points themselves to shorten the route and get rid of
    # points that are dead ends in the route
    best_points = optimize_points(
        gh, center, radius, points)

    path = get_path_from_points(gh, best_points)
    # show_path(path, best_points)
    write_gpx("out10.gpx", path)
