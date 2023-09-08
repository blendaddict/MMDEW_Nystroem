import numpy as np

from mmdew.abstract import RegionalDriftDetector, DriftDetector
from mmdew.bucket_stream2 import BucketStream
from mmdew.mmd import MMD

class MMDEWAdapter(DriftDetector):
    def __init__(self, gamma, alpha=.1):
        """
        :param gamma: The scale of the data
        :param alpha: alpha value for the hypothesis test
      
        """
        self.gamma=gamma
        self.alpha = alpha
        self.logger = None
        self.detector = BucketStream(gamma=self.gamma, alpha=self.alpha)
        self.element_count = 0
        super(MMDEWAdapter, self).__init__()

    def name(self) -> str:
        return "MMDEW" 

    def parameter_str(self) -> str:
        return r"$\alpha = {}$".format(self.alpha)

    def pre_train(self, data):
        # hier können wir estimate_gamma ausführen
        self.gamma = MMD.estimate_gamma(data)
        #print(f"gamma: {self.gamma}")
        self.detector = BucketStream(gamma=self.gamma , alpha=self.alpha)
    

    def add_element(self, input_value):
        """
        Add the new element and also perform change detection
        :param input_value: The new observation
        :return:
        """

        self.element_count+=1
        self.detected_cp = False
        prev_cps = len(self.detector.get_changepoints())
        bsstring = ""
        #for i in range(0, len(self.detector.buckets)):
        #   bsstring += "(" + str(self.detector.buckets[i].uncompressed_capacity) + " " + str(len(self.detector.buckets[i].weights)) + ")"
        #first cp of mnist is at 6825

        #mmdcalc = "undefined"
        #if(len(self.detector.buckets) > 2):
        #    mmdcalc = str(self.detector.mmd(1))

        #print(f"elements read: {self.element_count}, mmd between biggest bucket and rest: {mmdcalc} current bs stream length: {len(self.detector.buckets)} stream: {bsstring}")
        #print(f"mmd between biggest bucket and rest: {self.detector.mmd(1)} " )
        self.detector.insert(input_value[0])
        if len(self.detector.get_changepoints()) > prev_cps:
            self.delay = self.element_count - self.detector.get_changepoints()[-1]
            self.detected_cp = True
#            print("Detected")

    def detected_change(self):
        return self.detected_cp
    
    def metric(self):
        return 0

 
