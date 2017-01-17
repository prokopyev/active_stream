import threading
import logging
import numpy as np

import shared

class Classifier(threading.Thread):
    '''
    Classifies statuses as relevant / irrelevant based on classification model
    trained by `Trainer()` and placed into `queues['model']`.

    Appends to the status object a field 'classifier_relevant' containing a
    binary classification (bool) and a field 'probability_relevant' containing
    the probability this classification is based on.

    If a status is uncertain (predicted probability between `uncertain_low` and
    `uncertain_high`) it is put into queues['annotator'] for manual annotation by the oracle.

    Arguments:
    --------------- 
    queues: dict containing queues to pass data between threads.
    uncertain_low: Lower bound for annotation threshold.
    uncertain_high: Upper bound for annotation threshold.
    name: str, name of the thread.
    '''

    def __init__(self, queues, uncertain_low=0.4, uncertain_high=0.6, name=None):

        logging.debug('Initializing Classifier...')

        super(Classifier, self).__init__(name=name)

        self.clf = None
        self.queues = queues

        logging.debug('Success')

    def run(self):
        '''
        Run the thread
        '''

        logging.debug('Running.')
        while not shared.TERMINATE:
            # Check for new model
            if shared.RUN_TRAINER:
                X_labelled = []
                y = []
                cursor = self.queues['database'].find({'manual_relevant': {'$ne': None}})
                for d in cursor:
                    X_labelled.append(d['embedding'])
                    y.append(d['manual_relevant'])

                X_labelled = np.array(X_labelled)
                y = np.array(y)

                self.clf.partial_fit(X, y, classes=np.array([0, 1]))

                X = []
                probs = []
                cursor = self.queues['database'].find({}, {'embedding': 1})
                for d in cursor:
                    X.append(d['embedding'])

                X = np.array(X)
                probs = self.clf.predict_proba(X)
                entropy = stats.distributions.entropy(probs.T)

                max_entropy = np.argsort(entropy)[-10:]

                # find which status match those in max_entropy
                # and set those to have to_annotate = True if they aren't already
                # annotated
                # self.queues['database'].update({'id': })

        logging.debug('Terminating.')
        self.cleanup()

    def classify_status(self, status):
        '''
        Assess relevance of a status

        Takes a status and classifies it as relevant / irrelevant. And appends a
        predicted probability to the status object.

        Uses `self.clf` to classify the status. As long as no model has been
        trained it assignes 0.5 probability to all statuses.

        Arguments:
        --------------- 
        status: dict, a status (tweet) with additional fields ('embedding',
            'manual_relevant', 'classifier_relevant')
        '''

        if self.clf is None:
            prob = 0.5
        else:
            X = np.array(status['embedding']).reshape(1,-1)
            pred_prob = self.clf.predict_proba(X) 
            prob = pred_prob[0][1]

        status['probability_relevant'] = prob
        if prob < 0.5:
            status['classifier_relevant'] = False
        else:
            status['classifier_relevant'] = True
        return status
       

    def update_clf(self):
        '''
        Checks if there is a new clf (model) in `queues['model']` and if so
        updates the attribute `self.clf`
        '''
        if not self.queues['model'].empty():
            logging.debug('Acquiring Model')
            self.clf = self.queues['model'].get()

        return None

    def cleanup(self):
        return None

class Trainer(threading.Thread):
    '''
    (Re)Trains classification model.

    When `ONE_POSITIVE` and `ONE_NEGATIVE` and `RUN_TRAINER` are set to True (by
    `Annotator()`, (re-)train the model and put it into `queues['model']

    Arguments:
    --------------- 
    queues, dict containing all queues for passing data between threads (see
        main script)
    clf: A classifier object. Must contain a `fit(X, y)` method (see sk learn
        models)
    '''

    def __init__(self, queues, clf, name=None):
        logging.debug('Initializing Trainer...')
        super(Trainer, self).__init__(name=name)
        self.clf = clf
        self.queues = queues
        logging.debug('Success')

    def train_model(self):
        '''
        (Re)train the model on all annotated tweets in the db
        '''
        # Transform data y = []
        X = []
        y = []
        # Get all manually annotated docs from db
        cursor = self.queues['database'].find({'manual_relevant': {'$ne': None}}) 
        for d in cursor:
            X.append(d['embedding'])
            y.append(d['manual_relevant'])

        X = np.array(X)
        y = np.array(y)
        
        # Fit model
        self.clf.partial_fit(X, y, classes=np.array([0, 1]))

        # Pass model to classifier
        self.queues['model'].put(self.clf)
        shared.RUN_TRAINER = False
        
    def run(self):

        logging.debug('Running.')
        # Wait for first positive / negative annotation
        while not shared.TERMINATE:
        
            if not shared.ONE_POSITIVE or not shared.ONE_NEGATIVE:
                continue

            # After that run everytime prompted by the annotator thread
            if shared.RUN_TRAINER:
                logging.debug('Retraining Model...')
                self.train_model()
                logging.debug('Trained Model.')

        logging.debug('Terminating.')
        self.cleanup()

    def cleanup(self):
        return None
