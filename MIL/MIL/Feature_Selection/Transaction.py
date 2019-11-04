class Transaction(object):
    """description of class"""
    
    def __init__(self, *args):
        self.TransID = args[0][0]
        self.AccountID = args[0][1]
        self.DeviceID = args[0][2]
        self.TransDate = args[0][4]
        self.IP = args[0][6]
        self.Browser = args[0][13]
        self.Date = args[0][22]
        self.Year = args[0][23]
        self.Month = args[0][24]
        self.Day = args[0][25]
        self.Hour = args[0][26]
        self.Minute = args[0][27]
        self.Second = args[0][28]
      

   