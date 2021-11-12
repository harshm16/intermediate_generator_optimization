import neptune
import neptunecontrib.monitoring.skopt as sk_utils
import skopt
import main

run = neptune.init(project_qualified_name='mishraharsh169/loss-weight-optimize',api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI3MDcwNzRiYS02MjViLTQyYmItYmMwOS0zYzZmMzdhZGY0NWQifQ==")
neptune.create_experiment(name = 'loss_w_optimize', upload_source_files=['*.py'])


SPACE = [
    skopt.space.Real(0.1, 1, name='loss_weight', prior='uniform')]


@skopt.utils.use_named_args(SPACE)
def objective(**params):
    return -1.0 * main.train(params)


monitor = sk_utils.NeptuneMonitor()
results = skopt.forest_minimize(objective, SPACE, n_calls=100, n_random_starts=10, callback=[monitor])
sk_utils.log_results(results)

best_auc = -1.0 * results.fun
best_params = results.x

print('best result: ', best_auc)
print('best parameters: ', best_params)


neptune.stop()



