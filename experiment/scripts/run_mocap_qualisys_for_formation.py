#!/usr/bin/python3
import os
import sys
import asyncio
import socket
import time
import json
import time
import numpy as np
import transforms3d
import xml.etree.ElementTree as ET
import pkg_resources
import qtm
import pathmagic
with pathmagic.context():
    from UdpProtocol import UdpProtocol


def create_body_index(xml_string):
    """
    Extract a name to index dictionary from 6-DOF settings xml
    """
    xml = ET.fromstring(xml_string)
    body_to_index = {}
    for index, body in enumerate(xml.findall("*/Body/Name")):
        body_to_index[body.text.strip()] = index
    return body_to_index

def publisher_tcp_main(config_data: dict):
    """
    Create a TCP/IP socket to publish the states estimation.
    The following two lines show how to get config_data:
        json_file = open('mocap_config.json')
        config_data = json.load(json_file)
    """
    # IP for publisher
    HOST_TCP = config_data["QUALISYS"]["IP_STATES_ESTIMATION"]
    # Port for publisher
    PORT_TCP = int(config_data["QUALISYS"]["PORT_STATES_ESTIMATION"])

    server_address_tcp = (HOST_TCP, PORT_TCP)
    # Create a TCP/IP socket
    sock_tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    return sock_tcp, server_address_tcp

async def planner_udp_main(config_data: dict):
    """
    Create a UDP socket to publish the agent position to the planner.
    The following two lines show how to get config_data:
        json_file = open('mocap_config.json')
        config_data = json.load(json_file)
    """
    # IP for publisher
    HOST = config_data["QUALISYS"]["IP_STATES_ESTIMATION"]
    # Port for publisher
    PORT = int(config_data["QUALISYS"]["PORT_POSITION_PLANNER"])
    planner_address_udp = (HOST, PORT)

    # Create a UDP socket for data streaming
    loop = asyncio.get_running_loop()
    transport_planner, protocol_planner = await loop.create_datagram_endpoint(
        UdpProtocol, local_addr=None, remote_addr=planner_address_udp)
    return transport_planner, planner_address_udp

async def obs_udp_main(config_data: dict):
    """
    Create a UDP socket to publish the obstacle position to the planner.
    The following two lines show how to get config_data:
        json_file = open('mocap_config.json')
        config_data = json.load(json_file)
    """
    # IP for obstacle states publisher
    HOST_OBS = config_data["QUALISYS"]["IP_OBS_POSITION"]
    # Port for obstacle states publisher
    PORT_OBS = int(config_data["QUALISYS"]["PORT_OBS_POSITION"])
    server_address_obs = (HOST_OBS, PORT_OBS)
    # create a UDP streaming protocol for subscribing obstacles states
    loop_obs = asyncio.get_running_loop()
    transport_obs, protocol_obs = await loop_obs.create_datagram_endpoint(
        UdpProtocol, local_addr=None, remote_addr=server_address_obs)
    return transport_obs, server_address_obs

async def main(config_file_name):
    """ Main function """
    # Read the configuration from the json file
    json_file = open(config_file_name)
    config_data = json.load(json_file)

    # IP address for the mocap server
    IP_server = config_data["QUALISYS"]["IP_MOCAP_SERVER"]

    # Connect to qtm
    connection = await qtm.connect(IP_server)

    # Connection failed?
    if connection is None:
        print("Failed to connect")
        return

    # Take control of qtm, context manager will automatically release control after scope end
    async with qtm.TakeControl(connection, "password"):
        # await connection.new()
        pass

    # Get 6-DOF settings from QTM
    xml_string = await connection.get_parameters(parameters=["6d"])

    # parser for mocap rigid bodies indexing
    body_index = create_body_index(xml_string)
    # the one we want to access
    wanted_body = config_data["QUALISYS"]["NAME_SINGLE_BODY"]

    print("Rigid body is " + wanted_body)

    # Create a TCP/IP socket for states estimation streaming
    sock_tcp, server_address_tcp = publisher_tcp_main(config_data)
    # Bind the socket to the port
    sock_tcp.bind(server_address_tcp)

    # Create a UDP socket for streaming position to planner
    transport_planner, planner_address_udp = await planner_udp_main(config_data)

    obs_body = None

    if "NAME_OBSTACLE" in config_data["QUALISYS"].keys():
        # Create a UDP socket for streaming obstacle position to planner
        transport_obs, server_address_obs = await obs_udp_main(config_data)
        # define the obstacle body name
        obs_body = config_data["QUALISYS"]["NAME_OBSTACLE"]
        print("Obstacle is: ", obs_body)
    
    ########## comment this line for debugging
    # Listen for incoming connections
    sock_tcp.listen()

    # Wait for a connection
    print("waiting for a connection")
    print("You can execute src/run_mambo.py now.")
    print("You can execute the high-level planner now.")

    ########## comment this line for debugging
    connection_tcp, client_address = sock_tcp.accept()
    print("Built connection with", client_address)

    def on_packet(packet):
        # Get the 6-DOF data
        bodies = packet.get_6d()[1]

        if wanted_body is not None and wanted_body in body_index:
            # Extract one specific body
            t_now = time.time()
            wanted_index = body_index[wanted_body]
            position, rotation = bodies[wanted_index]

            ########## comment this line for debugging
            # send 6-DOF data via TCP/IP
            # concatenate the position and rotation matrix vertically
            msg_1 = np.asarray((position.x/1000.0, position.y/1000.0, position.z/1000.0) + rotation.matrix + (t_now, ), dtype=float).tobytes()
            connection_tcp.sendall(msg_1)

            # send position data to the planner via UDP
            msg_2 = np.asarray((position.x/1000.0, position.y/1000.0, position.z/1000.0), dtype=float).tobytes()
            transport_planner.sendto(msg_2, planner_address_udp)

            # print("position")
            # print([position.x/1000.0, position.y/1000.0, position.z/1000.0])
        else:
            raise Exception("There is no such a rigid body!")

        if obs_body is not None and obs_body in body_index:
            # Extract one specific body
            obs_index = body_index[obs_body]
            position, rotation = bodies[obs_index]
            # print(obs_body)

            # send obstacle positions to the planner via UDP
            msg_3 = np.asarray((position.x/1000.0, position.y/1000.0, position.z/1000.0), dtype=float).tobytes()
            transport_obs.sendto(msg_3, server_address_obs)

            # print(obs_body + " position: ", [round(position.x/1000.0,2), round(position.y/1000.0,2), round(position.z/1000.0,2)])

    # Start streaming frames
    # Make sure the component matches with the data fetch function, for example: packet.get_6d() with "6d"
    # Reference: https://qualisys.github.io/qualisys_python_sdk/index.html
    await connection.stream_frames(components=["6d"], on_packet=on_packet)


if __name__ == "__main__":
    # load mambo index from command line arguments
    if len(sys.argv) == 2:
        mambo_idx = sys.argv[1]
    else:
        mambo_idx = 1

    # load the configuration as a dictionary
    config_file_name = os.path.expanduser("~") + \
        "/github/Reinforcement-Learning-for-Multi-agent-Formation-Control/experiment/config_aimslab_" + \
        str(mambo_idx) + ".json"

    # config_file_name = os.path.expanduser("~") + \
    #     "/github/DrMaMP-Distributed-Real-time-Multi-agent-Mission-Planning-Algorithm/experiment/config_aimslab_ex_" + \
    #     str(mambo_idx) + ".json"

    # Run our asynchronous main function forever
    asyncio.ensure_future(main(config_file_name))
    asyncio.get_event_loop().run_forever()
