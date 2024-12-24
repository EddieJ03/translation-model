import * as pulumi from "@pulumi/pulumi";
import * as docker from "@pulumi/docker";
import * as k8s from "@pulumi/kubernetes";
import TraefikRoute from "./TraefikRoute";

// Load configuration
const config = new pulumi.Config();

const baseStack = new pulumi.StackReference(config.require("baseStackName"));

// Create Kubernetes provider
const provider = new k8s.Provider("k8s-provider", {
  kubeconfig: baseStack.requireOutput("kubeconfig"),
});

const dockerImage = new docker.Image("my-modelaugbbqezdqix-image", {
  imageName: `docker.io/${config.require(
    "dockerUsername"
  )}/my-modelaugbbqezdqix-image:latest`, // Use your Docker Hub username and desired image name
  build: {
    context: "../", // Path to your Dockerfile
    platform: "linux/amd64", // Target platform (optional)
  },
});

// Kubernetes Deployment
const appName = "my-modelaugbbqezdqix";
const appLabels = { appClass: appName };
const deployment = new k8s.apps.v1.Deployment(
  `${appName}-deployment`,
  {
    metadata: { labels: appLabels },
    spec: {
      replicas: 1,
      selector: { matchLabels: appLabels },
      template: {
        metadata: { 
          labels: appLabels,
          annotations: {
            "kubectl.kubernetes.io/restartedAt": new Date().toISOString(), // Dynamically set timestamp
          }, 
        },
        spec: {
          containers: [
            {
              name: appName,
              image: dockerImage.imageName,
              ports: [
                {
                  name: "http",
                  containerPort: parseInt(config.require("port")),
                },
              ],
              env: [
                { name: "LISTEN_PORT", value: config.require("port") },
                {
                  name: "MLFLOW_TRACKING_URI",
                  value: config.require("mlflowURI"),
                },
                { name: "MLFLOW_RUN_ID", value: config.require("runID") },
                {
                  name: "AZURE_STORAGE_CONNECTION_STRING",
                  value: config.requireSecret("azureStorageConnectionString"),
                },
              ],
            },
          ],
        },
      },
    },
  },
  { provider: provider }
);

// Create a Horizontal Pod Autoscaler
const hpa = new k8s.autoscaling.v2.HorizontalPodAutoscaler(
  "my-modelaugbbqezdqix-hpa",
  {
    spec: {
      scaleTargetRef: {
        apiVersion: "apps/v1",
        kind: "Deployment",
        name: deployment.metadata.name,
      },
      maxReplicas: 2,
      metrics: [
        {
          type: "Resource",
          resource: {
            name: "cpu",
            target: {
              type: "Utilization",
              averageUtilization: 50,
            },
          },
        },
      ],
    },
  },
  { provider }
);

const service = new k8s.core.v1.Service(
  `${appName}-service`,
  {
    metadata: { labels: appLabels },
    spec: {
      type: "ClusterIP",
      ports: [{ port: parseInt(config.require("port")), targetPort: "http" }],
      selector: appLabels,
    },
  },
  { provider: provider }
);

new TraefikRoute(
  "my-modelaugbbqezdqix",
  {
    prefix: "/models/my-modelaugbbqezdqix",
    service,
    namespace: "default",
  },
  { provider: provider, dependsOn: [service] }
);

// Outputs
export const deploymentName = deployment.metadata.name;
export const serviceName = service.metadata.name;
