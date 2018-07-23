import vtk
import numpy as np
import visualizeTools3D as vis
import csv


videotoolTransforms = np.load('toolTransforms.npy')
videoPtCloud = np.load('toolPtCloud.npy')

#axes
axes = vtk.vtkAxesActor()

# Render Window
renderWindow = vtk.vtkRenderWindow()
renderWindow.SetSize(1280, 960)

#camera
camera = vtk.vtkCamera()
camera.SetPosition(0, 50, -50)
camera.SetFocalPoint(0, 0, 1)
camera.SetViewUp(0, 1, 0)
camera.SetClippingRange(0, 150)
camera.SetViewAngle(75)
camera.Modified()

# Renderer
renderer = vtk.vtkRenderer()
renderer.AddActor(axes)
renderer.SetBackground(0.5, 0.5, 0.5)
renderer.SetActiveCamera(camera)
renderWindow.AddRenderer(renderer)

# Interactor
renderWindowInteractor = vtk.vtkRenderWindowInteractor()
renderWindowInteractor.SetRenderWindow(renderWindow)
renderWindowInteractor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
renderWindowInteractor.Initialize()


#record video

writer = vtk.vtkAVIWriter()
writer.SetRate(30)
writer.SetFileName("test.avi")

# Sign up to receive TimerEvent
cb = vis.AddToolsTimerCallBack(renderer, videotoolTransforms, videoPtCloud
                               , writer)
renderer.AddActor(cb.tool1Actor)
renderer.AddActor(cb.tool2Actor)




renderWindowInteractor.AddObserver('TimerEvent', cb.execute)
timerId = renderWindowInteractor.CreateRepeatingTimer(200)
vis.AddToolsTimerCallBack.timerId = timerId
#cb.setActors(tool1.vtkActor, tool2.vtkActor)
# cb.transform1 = np.identity(4)
# cb.transform2 = np.identity(4)

renderWindowInteractor.Start()


