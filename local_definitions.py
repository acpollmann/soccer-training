def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()



# Given a vector of accuracy ("score") values for different numbers of features, print the maximum
def print_max_accuracy(scores, mode='testing'):
    import numpy as np
    print(f"Max score in {mode}: = {np.round(np.max(scores), 3)} with {np.argmax(scores) + 1} features")



# Make a plot of the training accuracy vs. number of features using ploty. Overlay the training accuracies
# and the testing accuracies in the same plot.
def plot_accuracy(xdata, train_scores, test_scores, clf_name='Random Forest',
                  xaxis_title='number of features', yaxis_title='accuracy', print_max=True):

    import plotly.graph_objects as go

    fig = go.Figure(go.Scatter(
        x=xdata,
        y=train_scores,
        name ="Training accuracy w/ CV"))
    fig.add_trace(go.Scatter(
        x=xdata,
        y=test_scores,
        name="Testing accuracy"))

    fig.update_layout(title=f"Training and testing accuracies for {clf_name}",
                      xaxis_title='number of features',
                      yaxis_title='accuracy')
    fig.show()

    if print_max:
        print_max_accuracy(train_scores, mode='training')
        print_max_accuracy(test_scores, mode='testing')



def plot_accuracy_with_errorbar(xdata, train_scores, train_err, test_scores, test_err, clf_name='Random Forest',
                  xaxis_title='number of features', yaxis_title='accuracy', print_max=True):

    import plotly.graph_objects as go

    fig = go.Figure(go.Scatter(
        x=xdata,
        y=train_scores,
        error_y=dict(
            type='data',
            array=train_err,
            visible=True),
        name ="Training accuracy w/ CV"
        ))
    fig.add_trace(go.Scatter(
        x=xdata,
        y=test_scores,
        error_y=dict(
            type='data',
            array=test_err,
            visible=True),
        name="Testing accuracy"))

    fig.update_layout(title=f"Training and testing accuracies for {clf_name}",
                      xaxis_title='number of features',
                      yaxis_title='accuracy')
    fig.show()

    if print_max:
        print_max_accuracy(train_scores, mode='training')
        print_max_accuracy(test_scores, mode='testing')