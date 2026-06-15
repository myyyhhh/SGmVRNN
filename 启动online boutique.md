## 监控




minikube service grafana -n monitoring

name admin

password admin



kubectl port-forward svc/prometheus -n monitoring 9090:9090