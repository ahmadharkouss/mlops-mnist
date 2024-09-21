# MNIST API

## Build the Docker image
```bash
docker build -t mnist-api .
```

## Run the Docker container
```bash
docker run -p 7000:7000 mnist-api
```

## Access the API documentation
Go to the following page: http://0.0.0.0:7000/docs <br>
Click on 'Try it out', load an image and then click on 'Execute' to test the API. <br>
You can find test images in the 'test_images' folder.

## Stop the Docker container
```bash
docker stop $(docker ps -a -q --filter ancestor=mnist-api --format="{{.ID}}")
```
Main branch deploy directly to https://mnist.devgains.com
```