# Test for Brilliant 

import sys
import time
import nmengine
import ctypes

#
# Trains the neuromem network
#
def tc_learn(target, cat, vector , v_size): 
    
    learn_req = nmengine.LearnReq()
    
    # Send max length to avoid gabages in neuron memory. 
    # If you set the size with actual vector length (< max), 
    # the remaining value will be filled with garbege value
    learn_req.size = nmengine.NEURON_MEMORY
    learn_req.vector[:len(vector)] = vector
    learn_req.category = cat
    # This is optional field for retrieving affected neuron infromation
    # If you set the value to 1, it takes more time to learn.
    learn_req.query_affected = 1
    print('\nlearn req(cat, affected flag, len)', learn_req.category, learn_req.query_affected, learn_req.size, tuple(learn_req.vector))
    print('\ntrain a vector to network', nmengine.learn(target, learn_req))

    # Prints affected neurons
    for i in range(learn_req.affected_count):
        neuron = learn_req.affected_neurons[i]
        print('\nAffected neurons (nid, cat, aif) :', neuron.nid, neuron.cat, neuron.aif)


#
# Classify input vector 
#
def tc_classify(target, vector, k): 
    
    classify_req = nmengine.ClassifyReq()
    classify_req.size = nmengine.NEURON_MEMORY
    classify_req.vector[:len(vector)] = vector
    classify_req.k = k

    nmengine.classify(target, classify_req)
    print('\nnetwork status', classify_req.status, 'matched:', classify_req.matched_count)
    # Prints matched neurons
    for i in range(classify_req.matched_count):
        print('\nMatched (nid, dis, cat, degen) :', classify_req.nid[i], classify_req.distance[i], classify_req.category[i], classify_req.degenerated[i] )
       


# Set the library path directly if library can not load 
# nmengine.set_lib_path('/usr/local/lib/')

# Gets information of devices attached
result, devices = nmengine.get_devices()
# Stop if there is no detccted device.
if result == nmengine.ERROR_DEVICE_NOT_FOUND \
    or len(devices) == 0:
    print('no device found.', result)
    sys.exit()
elif result != nmengine.SUCCESS:
    print('failed to get device list.', result)
    sys.exit()

count = len(devices)
print(count, 'device(s) found')
for i in range(count):
    print('id:', devices[i].id, "device(", devices[i].vid, devices[i].pid, \
        devices[i].version, ")")
print()


# Select a device to use.
target = devices[0]


# Connects to target device
print("connect to device", target.id, ",", nmengine.connect(target))
print()


# Gets device firmware version
result, version = nmengine.get_version(target)
print('check device firmware version', version)
print()


# Gets network information
# Notice, this function performs the initialization of the NM500
# Do not use this function during learning/classfying
network_info = nmengine.NetworkInfo()
nmengine.get_network_info(target, network_info)
print('get network info', network_info.neuron_count, network_info.neuron_memory_size, network_info.version)
print()


# Gets the number of neurons used(committed)
result, count = nmengine.get_neuron_count(target)
print('the number of neurons used', count)
print()


# Resets target device (the +error code:108+ is general result here)
# It initializes the device and clear the neuromem network.
# The context and min/max influence fields will be set to the default value (0x01, 0x0002, 0x4000) 
# Also, you can use "nmengine.forget(target))", it is faster than reset()
# further information, please refer to api document.
print('reset device', nmengine.forget(target))


LENGTH=4
bytearray = ctypes.c_int * LENGTH

vector1=bytearray()
vector1[0]=100
vector1[1]=0
vector1[2]=0

tc_learn(target, 1, vector1, LENGTH)

vector2=bytearray()
vector2[0]=0
vector2[1]=100
vector2[2]=0

tc_learn(target, 2, vector2, LENGTH)

vector3=bytearray()
vector3[0]=0
vector3[1]=0
vector3[2]=100

tc_learn(target, 3, vector3, LENGTH)


# initializing the variables to hold the result of a recognition
# with up to k responses of the top firing neurons, if applicable
#time.sleep(10)
print("\n**************************************************\n")  
k=3
bytearray = ctypes.c_int * k
dists=bytearray()
cats=bytearray()
nids=bytearray()

print("\nVector (90,50,0)")
vector4=bytearray()
vector4[0]=90
vector4[1]=50
vector4[2]=0

tc_classify(target, vector4, k)

#time.sleep(10)

print("\nVector (50,90,0)")  
vector5=bytearray()
vector5[0]=50
vector5[1]=90
vector5[2]=0
tc_classify(target, vector5, k)

#time.sleep(10)
print("\nVector (100,100,100)")

vector6=bytearray()
vector6[0]=100
vector6[1]=100
vector6[2]=100

tc_classify(target, vector6, k)

