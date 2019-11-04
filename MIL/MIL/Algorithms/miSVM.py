import numpy as np
import sklearn.metrics
import sklearn.svm

def create_bags_from_instance_ids(instance_samples, instance_targets, instance_bag_ids):
	num_of_instances = len(instance_samples)
	bag_samples = []
	bag_targets = []
	bag_instance_targets = []
	for bag_id in np.unique(instance_bag_ids):
		this_bag_instance_indices = filter(lambda instance_index: instance_bag_ids[instance_index] == bag_id, range(0, num_of_instances))
		this_bag_instances = [instance_samples[i] for i in this_bag_instance_indices]
		this_bag_instance_targets = [instance_targets[i] for i in this_bag_instance_indices]
		bag_samples.append(this_bag_instances)
		bag_instance_targets.append(this_bag_instance_targets)
		bag_targets.append(1 if np.max(this_bag_instance_targets) == 1 else -1)
	return bag_samples, bag_targets, bag_instance_targets

class MultipleInstanceSVC(object):
	
    def __init__(self, **kwargs):
        self.svc_parameters = kwargs
        self.svc_parameters['probability'] = True
        self.svc = None
        self.num_of_fit_iterations_ = 0
    
    def fit(self, X, y, max_iterations=100, K=None, per_iter_func=None, **kwargs):
        bag_level_samples = X
        bag_level_targets = y
        kernel = K
        instance_level_samples = []
        instance_level_targets = []
        bag_instance_index_lists = []
        instance_index = 0
        for bag_level_sample, bag_level_target in zip(bag_level_samples, bag_level_targets):
            bags_instance_level_samples = bag_level_sample
            instance_level_samples.extend(bags_instance_level_samples)
            instance_level_targets.extend([bag_level_target] * len(bags_instance_level_samples))
            bag_instance_indices = range(instance_index, instance_index + len(bags_instance_level_samples))
            bag_instance_index_lists.append(bag_instance_indices)
            instance_index += len(bags_instance_level_samples)
            
        instance_level_samples = np.array(instance_level_samples)
        instance_level_targets = np.array(instance_level_targets)
        iteration_instance_level_targets = np.copy(instance_level_targets)
        self.svc = None
        self.num_of_fit_iterations_ = 0
        if not kernel == None:
            subset_kernel = kernel[instance_level_samples, :][:, instance_level_samples]
            
            while True:
                if not per_iter_func == None:
                    per_iter_func(self.num_of_fit_iterations_, iteration_instance_level_targets)
                
                self.svc = sklearn.svm.SVC(**self.svc_parameters)
                if kernel == None:
                    self.svc.fit(instance_level_samples, iteration_instance_level_targets, **kwargs)
                    instance_level_labels = self.svc.predict(instance_level_samples)
                    instance_level_probabilities = self.svc.predict_proba(instance_level_samples)
                else:
                    self.precomputed_kernel_training_indices_ = instance_level_samples
                    self.svc.fit(subset_kernel, iteration_instance_level_targets, **kwargs)
                    instance_level_labels = self.svc.predict(subset_kernel)
                    instance_level_probabilities = self.svc.predict_proba(subset_kernel)
                
                for bag_index, bag_instance_indices in enumerate(bag_instance_index_lists):
                    bag_level_target = bag_level_targets[bag_index]
                    if bag_level_target == -1:
                        instance_level_labels[bag_instance_indices] = -1
                    else:
                        if np.sum(np.clip(instance_level_labels[bag_instance_indices], 0, 1)) == 0:
                            max_probability_index = np.argmax(instance_level_probabilities[:, 1])
                            instance_level_labels[max_probability_index] = 1
                if np.sum(np.abs(iteration_instance_level_targets - instance_level_labels)) == 0 or self.num_of_fit_iterations_ >= max_iterations:
                    break
                else:
                    iteration_instance_level_targets = instance_level_labels
                    del self.svc
                self.num_of_fit_iterations_ += 1
        return self
        

    def predict_in_bag(self, X, K=None):
        bag_level_samples = X
        bag_level_labels = []
        for bag_level_sample in bag_level_samples:
            bags_instance_level_samples = bag_level_sample
            if K == None:
                bags_instance_level_labels = self.svc.predict(bags_instance_level_samples)
            else:
                subset_kernel = K[bags_instance_level_samples, :][:, self.precomputed_kernel_training_indices_]
                bags_instance_level_labels = self.svc.predict(subset_kernel)
                bag_level_labels.append(bags_instance_level_labels)
        return bag_level_labels

    def predict(self, X, K=None):
        bag_instance_labels = self.predict_in_bags(X, K=K)
        return [max(bags_instance_level_labels) for bags_instance_level_labels in bag_instance_labels]
    
    def predict_instances(self, X, K=None):
        if K == None:
            return self.svc.predict(X)
        else:
            subset_kernel = K[X, :][:, self.precomputed_kernel_training_indices_]
            return self.svc.predict(subset_kernel)

    def predict_proba_in_bag(self, X, K=None):
        bag_level_samples = X
        bag_level_probas = []
        for bag_level_sample in bag_level_samples:
            bags_instance_level_samples = bag_level_sample
            if K == None:
                bags_instance_level_probas = self.svc.predict_proba(bags_instance_level_samples)
            else:
                subset_kernel = K[bags_instance_level_samples, :][:, self.precomputed_kernel_training_indices_]
                bags_instance_level_probas = self.svc.predict_proba(subset_kernel)
                bag_level_probas.append(bags_instance_level_probas)
        return bag_level_probas

    def predict_proba_instances(self, X, K=None):
        if K == None:
            return self.svc.predict_proba(X)
        else:
            subset_kernel = K[X, :][:, self.precomputed_kernel_training_indices_]
            return self.svc.predict_proba(subset_kernel)
    
    def score_in_bag(self, X, y, K=None):
        bag_level_samples = X
        bag_level_targets = y
        bag_level_scores = []
        for bag_level_sample, bag_level_target in zip(bag_level_samples, bag_level_targets):
            bags_instance_level_samples = bag_level_sample
            bags_instance_level_targets = bag_level_target
            bags_instance_level_labels = self.predict_instances(bags_instance_level_samples, K=K)
            bags_instance_level_scores = sklearn.metrics.accuracy_score(bags_instance_level_labels, bags_instance_level_targets)
            bag_level_scores.append(bags_instance_level_scores)
        return bag_level_scores

    def score(self, X, y, K=None):
        return sklearn.metrics.accuracy_score(self.predict(X, K=K), y)

    def score_instances(self, X, y, K=None):
        return sklearn.metrics.accuracy_score(self.predict_instances(X, K=K), y)
