import logging
import json

class Variables:
#    CRITICAL 	50
#    ERROR 	    40
#    WARNING 	30
#    INFO 	    20
#    DEBUG 	    10
#    NOTSET 	0
    debugLvl = 10
    FORMAT = '%(asctime)s %(levelname)-10s %(filename)-30s %(message)s '
    logging.basicConfig(format=FORMAT, level=30, datefmt='%Y-%m-%d %H:%M:%S',)
    logger = logging.getLogger("base")
    logger.setLevel(debugLvl)
    with open('data/relations.txt','r') as inf:
        labelsDict = json.loads(inf.read())
        labelsDictInverted = dict(map(reversed, labelsDict.items()))
