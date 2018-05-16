import VBTrackingDevice
import VBStereoCalibration
import zmq

# load from file
calibParameters = VBStereoCalibration.StereoCalibration('', False, False, True)

tracking = VBTrackingDevice.VBTrackingDevice()
tracking.initialize(calibParameters.StereoParameters)

run = True
while(run):
    #read from zeromq

    tracking.track()
    transforms = tracking.getTransformations()
    print(transforms)

