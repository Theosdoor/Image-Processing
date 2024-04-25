# Image Processing Coursework

## How to use
By default, the images inside image_processing_files/xray_images are processed. To specify a directory of images to process, inside project's directory run
```
python3 main.py unseen_test_images
```.

If you want to start the processing partway through the image directory, set `start_from` variable at bottom of main.py to name of desired start point in directory. 

Note that the Criminisi_Inpainter class can take two extra arguments: *verbose = True/False*, which updates the user on inpainting progress, and *show_progress = True/False* which saves an image of the current inpainting progress in the parent directory each iteration.

## References
See pdf for citations.