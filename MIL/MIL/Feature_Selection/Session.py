class Session(object):
    """description of class"""

    def __init__(self, *args):
        self.TransIDs = args[0]
        self.Interval = args[1]
        self.Frequency = args[2]
        self.Target = args[3]

    