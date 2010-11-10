import mlpython.misc.io as mlio
import numpy as np
import os

def load(dir_path,set_id=0,load_to_memory=False,dtype=np.float64):
    """
    Loads the Yahoo! Learning to Rank Challenge data.

    The data is given by a dictionary mapping from strings
    'train', 'valid' and 'test' to the associated pair of data and metadata.
    
    Option 'set_id' determines the set that is loaded (0, 1 or 2, default is 0).

    Set 0 is a "home made" train/valid split of the original training set, 
    required since only the training set is labeled (until all the data is
    released). Not test set is generated for that purpose.

    Note that, because the data is quite big, it is not loaded in memory and is
    instead always read directly from the associated files.

    Defined metadata: 
    - 'input_size'
    - 'scores'
    - 'n_queries'
    - 'length'

    """
    
    input_size=700
    dir_path = os.path.expanduser(dir_path)
    sparse=False

    def convert(feature,value):
        if feature != 'qid':
            raise ValueError('Unexpected feature')
        return int(value)

    def load_line(line):
        return mlio.libsvm_load_line(line,convert,int,sparse,input_size)

    if set_id == 0:
        n_queries = [16951,2993]
        lengths = [402167,70967]
        #lengths = [294336,178798]

        train_file,valid_file = [os.path.join(dir_path, 'set1.' + ds + '.txt') for ds in ['in_house_train','in_house_valid']]
        # Get data
        train,valid = [mlio.load_from_file(f,load_line) for f in [train_file,valid_file]]

        if load_to_memory:
            train,valid = [mlio.MemoryDataset(d,[(input_size,),(1,),(1,)],[dtype,int],l) for d,l in zip([train,valid],lengths)]

        # Get metadata
        train_meta,valid_meta = [{'input_size':input_size,
                                  'scores':range(5),
                                  'n_queries':nq,
                                  'length':l,
                                  'n_pairs':l} for nq,l in zip(n_queries,lengths)]

        return {'train':(train,train_meta),'valid':(valid,valid_meta)}
    else:
        if set_id == 1:
            n_queries = [19944,2994,6983]
            lengths = [473134,71083,165660]
        else:
            n_queries = [1266,1266,3798]
            lengths = [34815,34881,103174]

        # Get data file paths
        train_file,valid_file,test_file = [os.path.join(dir_path, 'set' + str(set_id) + '.' + ds + '.txt') for ds in ['train','valid','test']]
        # Get data
        train,valid,test = [mlio.load_from_file(f,load_line) for f in [train_file,valid_file,test_file]]
        if load_to_memory:
            train,valid,test = [mlio.MemoryDataset(d,[(input_size,),(1,),(1,)],[dtype,int],l) for d,l in zip([train,valid,test],lengths)]

        train_meta,valid_meta,test_meta = [{'input_size':input_size,
                                            'scores':range(5),
                                            'n_queries':nq,
                                            'length':l} for nq,l in zip(n_queries,lengths)]

        return {'train':(train,train_meta),'valid':(valid,valid_meta),'test':(test,test_meta)}

def obtain(dir_path):

    dir_path = os.path.expanduser(dir_path)
    train_file = os.path.join(dir_path, 'set1.train.txt')
    try:
        file = open(train_file)
        n_queries = 0
        in_house_train_file = os.path.join(dir_path, 'set1.in_house_train.txt')
        in_house_valid_file = os.path.join(dir_path, 'set1.in_house_valid.txt')
        train_file = open(in_house_train_file,'w')
        valid_file = open(in_house_valid_file,'w')
	# qids in validation set (sorry for the ridiculously long line...)
        qids_valid = [7,5,6,25,22,35,13,9,31,29,51,28,12,79,2,48,4,53,63,64,80,10,23,56,27,16,689,30,83,58,38,90,52,313,60,14,98,19,76,92,109,120,86,40,21,88,70,32,105,33,24,44,37,75,126,45,49,115,122,106,11,129,507,50,61,133,584,139,153,146,73,84,81,62,18,101,15,96,20,77,138,17,123,68,69,201,74,36,119,110,54,41,137,111,134,165,136,140,85,125,141,199,152,156,178,59,89,100,204,107,205,93,116,148,34,161,94,26,72,231,155,151,351,103,112,97,124,210,216,228,157,162,108,179,113,87,39,42,158,135,55,168,229,180,1024,1142,1016,985,1069,1074,1025,1267,1033,1029,1003,1190,1022,1017,1407,1124,1032,980,974,965,977,585,1276,1185,564,1198,128,1203,65,184,341,99,1018,91,350,969,114,998,966,981,1301,975,1351,1040,224,1027,1039,1055,968,967,1191,3,1049,976,994,1079,521,996,1070,1060,1013,970,979,1000,1052,1062,1303,1007,1085,984,1250,987,1353,1042,1026,1378,1379,1160,986,1071,1100,1076,1125,1333,71,1095,1008,1279,1114,1494,1261,1218,1135,1063,1031,1058,983,1068,991,1147,1094,1030,1090,1059,1082,1088,982,1409,1441,1138,988,1382,1157,1064,1232,995,1072,1080,1012,43,989,1285,1023,1247,173,1046,1174,1362,1051,1119,1104,1145,185,1019,1195,1253,1462,1274,1412,1099,1134,1509,992,1180,1108,1385,990,1291,1207,1073,1161,1277,603,1015,1222,1239,1310,1001,1057,1020,1041,1006,1035,1075,1092,1101,613,1312,235,1096,1083,1237,1103,1146,1077,1242,1048,1338,1187,1110,1406,1573,1507,1148,1350,1053,971,1115,1011,1325,1254,1268,1519,1113,1044,1136,57,1214,1295,1154,1165,1192,1098,1259,203,1210,1305,1368,1126,999,1384,1036,1156,1220,1294,1537,1307,1117,1370,1569,1585,1245,1212,2586,1564,1233,1067,1408,1416,1133,1421,1093,1390,1178,1229,1640,1065,1330,1584,1430,1196,1641,1128,1194,1021,1004,1336,1151,1189,1429,1324,1043,1182,736,1664,171,1131,1530,480,218,779,1568,1265,363,268,1571,1213,1413,1545,1598,1465,1614,1680,1184,1376,1111,1326,1197,1144,1230,1599,1399,1186,1262,1273,1436,1657,1554,164,1097,1246,1457,1206,1389,1790,1593,1348,1283,1102,1745,1644,1240,1334,1252,1050,973,1570,1647,1028,1176,1127,1066,1290,1014,1403,1313,1056,1293,1447,1321,1356,1369,1297,1454,1359,1343,1415,1167,1231,1081,1566,1600,1282,1371,1470,1270,1451,1373,67,1159,1,78,588,1422,1091,1395,1445,1120,1296,1318,1179,1084,1381,1688,1587,234,1152,1236,1723,1469,1589,1286,1488,978,1271,1588,1327,1675,1037,1655,1316,1292,1377,1580,645,1681,1617,1005,1331,599,1304,490,1503,1202,1420,1685,1826,1089,1615,1643,1964,1432,1728,1401,1317,1116,1164,131,147,608,226,159,1129,47,236,225,1140,1177,255,1714,66,163,739,1518,773,1582,82,237,102,485,170,121,104,117,249,288,149,166,160,297,181,206,1188,1729,1456,264,1419,1474,1374,1962,1288,1602,1397,1499,1339,1166,1696,1425,1410,1757,1523,1534,1200,1266,1556,1749,1398,1209,1689,1223,266,2057,1744,1342,1697,2197,1216,791,1314,270,1611,1461,861,1607,1725,1105,1162,1238,1526,1248,1396,1337,1603,4744,1768,1361,1546,1848,1624,1704,1492,1726,127,1551,1559,1651,272,2258,837,1300,1009,859,240,1781,1844,1632,1711,1594,1733,1981,1256,1649,1512,1627,1365,1730,1402,1275,1660,1423,1479,1383,1473,1171,1493,1524,1442,1344,1249,1516,1411,1514,4773,1502,1322,1672,1575,1642,1633,1583,1309,150,1747,1773,1852,1718,2270,1489,1264,1665,1463,1505,1645,1527,1320,1345,1557,1106,1742,1346,1610,1251,1404,1391,1686,1437,1414,2010,1741,1086,1535,1298,1319,1168,1501,1221,1431,1010,1287,1529,1455,1622,1487,1272,1669,1281,2398,1983,1634,1280,1284,1531,1619,1727,207,144,223,167,172,2352,907,1183,299,600,189,175,154,1629,196,302,186,1586,1132,177,238,190,320,1130,1149,307,118,1700,188,334,239,194,183,1620,143,312,245,200,347,1595,326,2059,1219,261,356,132,2085,202,2011,276,267,284,130,277,253,2290,1625,333,1332,328,256,263,923,209,2345,342,142,641,2399,230,308,2721,2048,265,346,192,281,195,355,217,1754,250,282,176,657,1659,222,1217,244,382,324,841,358,390,401,1173,208,1707,317,336,354,280,2074,443,1141,337,283,278,452,2363,191,2731,258,287,197,1475,2799,454,289,269,211,271,367,368,364,338,212,1855,372,845,542,348,369,2098,1224,553,380,1761,359,1694,370,398,1155,2051,1163,375,373,275,290,309,430,885,291,215,379,304,305,391,520,220,330,393,1532,2500,1800,323,469,187,1199,385,340,534,388,449,331,399,374,2618,227,561,145,169,1639,292,198,649,458,402,482,174,394,219,1204,413,349,182,293,1405,221,1735,404,730,213,408,2625,494,232,259,909,656,303,241,628,233,314,1766,242,327,740,389,2878,518,587,311,410,505,252,254,1779,416,419,257,260,421,509,262,353,517,420,366,1847,246,522,273,429,525,285,360,439,247,318,677,1169,357,1311,423,455,361,424,295,604,562,457,432,387,543,1139,1153,362,248,444,477,400,300,332,274,343,1181,3088,1738,396,193,8,243,279,286,46,95,344,594,1882,792,506,1107,814,1087,407,428,523,1522,296,634,451,450,470,456,294,1452,484,467,1323,491,298,2099,395,500,321,1563,536,526,1854,541,329,578,310,412,1999,214,415,644,435,335,661,574,1978,1893,745,597,442,605,545,2083,392,464,486,365,620,479,554,766,488,496,1731,2726,629,301,3236,639,819,489,504,2096,576,643,2230,862,319,3267,3716,589,2097,609,386,953,627,3693,322,638,646,251,352,306,384,556,2797,577,640,3754,872,3767,590,498,956,499,703,417,647,425,653,595,727,625,1227,665,3352,728,2036,764,1257,670,793,511,2927,2839,674,630,3805,808,3808,3810,3218,835,652,532,1548,876,3832,2293,704,715,709,3838,782,821,851,3735,911,440,3839,3046,890,568,891,914,3734,3843,900,575,726,461,1054,4473,4999,5002,2094,3749,3615,4368,3295,940,3641,3686,3237,3704,3760,731,593,5263,5283,3752,3762,3667,5547,955,2234,958,962,463,3776,3737,2803,3792,5796,3770,2877,5827,3806,3300,3841,3281,4801,4833,755,5989,6070,3771,1226,3644,3753,770,4844,4859,4982,3782,3797,1710,781,3720,3813,3835,3755,2232,2410,2882,3948,4091,5176,3793,3756,3739,4791,6089,3829,3614,4930,4995,6171,3814,5052,3674,5240,5862,6109,6205,6241,426,6610,6619,4643,418,3761,3786,6299,4676,5901,5972,3769,5411,6310,3802,4717,6151,6334,3736,6322,3809,4460,811,5211,6340,371,4843,602,3740,3744,6339,6369,824,830,4367,6404,6477,6514,6478,3772,3745,315,5425,6615,465,3748,6621,848,3834,5058,3787,4559,617,5569,4692,3794,4703,3795,3796,3758,852,5633,651,3801,3759,5837,5210,6616,6014,6623,6626,6645,5858,6219,3765,3815,3799,466,3822,6635,6625,3820,6274,5640,6636,6650,5762,6640,3823,6653,865,6495,6641,6097,6665,6658,3844,6582,3975,5104,6604,874,6411,6646,6661,6678,1387,667,5114,6686,6687,669,6685,3824,6618,6627,6654,6699,6663,6701,6714,6720,6671,5166,6638,2076,468,6723,5826,6643,6741,4845,6768,6655,6657,6426,6775,6674,6689,6703,6263,6717,5935,6776,3836,6695,6752,3574,5248,5280,5541,6691,6718,6726,6785,6736,6790,6743,6611,6754,6622,6705,6624,6773,6783,6791,6792,6694,6801,6858,6235,671,6807,3840,6700,6722,6617,3842,6628,3915,5891,6774,5946,6808,6875,6122,4587,5938,6809,5208,6814,6702,6821,6179,5378,672,6213,6273,6293,6693,6826,6585,6782,5797,5811,676,6696,6901,6709,6827,6711,5828,3616,6728,6613,697,6734,5832,6917,6855,6946,6860,7001,7002,422,7061,7075,6802,6739,6861,6833,6871,6749,6750,6873,6891,6614,6913,6877,6319,6932,6935,7107,5885,7112,6944,6349,7122,6915,6374,6957,6753,6964,6966,6967,6377,6732,6755,6733,6764,7137,6930,7171,6950,5909,6757,7191,3722,6397,6760,6631,5970,6982,6761,6955,7192,6766,700,7010,6777,6788,6786,6978,7000,7040,6427,7217,7050,6535,6565,6666,6797,7052,7219,6772,6800,6581,7060,6597,6805,6838,7062,7220,7221,6845,7030,7083,7129,6620,5999,6881,6812,7226,7241,7273,3742,7031,7036,7139,6068,6924,7056,7147,7348,6796,6832,6633,7150,6841,7160,6676,3751,6075,6117,7374,7074,7178,3764,7185,7085,6126,7096,7006,7281,7409,6840,6883,6864,3766,7298,6163,6642,7412,6865,3773,6659,7109,548,6664,3778,7114,7415,6884,3788,7047,508,6172,701,6668,6679,7324,6869,6683,6253,7127,7170,6281,705,6312,6321,3798,6870,7332,3800,3804,7431,7459,7462,6710,6888,6895,519,6897,7190,7466,7337,6325,6712,6902,708,7493,6721,6681,6724,3812,3830,4834,6688,7356,7201,6355,535,7225,4840,6399,7053,6729,6904,7234,6499,316,4866,6725,4890,6629,6737,7390,6906,4897,4941,710,550,7071,5007,5009,7403,7502,6952,714,7432,6962,6745,7508,6926,483,6968,6527,7235,6630,5055,7458,5059,427,7236,7530,7522,5064,6934,6634,5102,5187,6639,6644,433,7239,6746,7242,6747,7154,7537,570,716,723,6751,6677,7230,747,5221,6938,7572,7259,7539,6763,5274,5290,6765,6648,5329,6874,6649,586,5332,7265,5559,6769,7552,601,7271,615,5618,7282,7588,6780,6793,6947,7585,6656,6660,6794,6804,6810,626,5763,6969,6971,6667,7312,7315,6984,6989,7594,7331,6669,6670,434,752,5775,6992,7360,753,7612,7611,767,6675,802,7359,7365,7372,5791,7385,6682,7638,6811,6824,6684,5798,7630,7666,6880,6825,7393,7668,6842,631,6999,7004,5844,7020,6843,815,6847,7645,438,7670,6690,6692,7673,6852,831,555,7005,632,7689,7016,7395,839,7028,7438,843,453,6853,6697,850,7032,7647,855,6704,5846,7407,7650,7445,7037,7679,7450,6713,6716,7693,6856,858,871,6866,6872,6929,7705,877,6907,7680,6910,7453,7484,6727,6742,887,7049,7696,5856,6748,6759,650,7717,6767,7068,7729,7077,7095,7710,663,7734,6770,6912,6779,7108,6988,473,7457,972,5921,6781,6803,7117,6914,5944,7768,5953,6815,7465,6816,7121,5959,6922,5967,6818,474,6819,6925,7123,6823,6937,6942,7722,6828,7474,6945,7742,7497,6831,7752,5969,7794,325,7126,6836,5996,7475,7481,6953,6960,7760,6963,7496,7769,7128,7131,7134,6839,7144,6848,696,3282,6973,6974,7033,7516,6024,6876,6882,6975,6027,7509,7812,7524,7149,6990,6887,6890,7009,6893,7545,6898,7819,7822,6899,7151,734,7015,7024,7025,7547,7548,3675,3678,377,6900,7550,339,7153,7041,6909,7044,7828,7054,7831,3774,7048,7051,7065,6918,7076,7772,7851,7133,7556,7773,6919,7590,7776,7079,6921,7591,6936,3779,7778,7086,7159,7592,7097,7605,6941,7779,345,624,7614,6093,3784,6951,7167,6104,7878,7691,3789,7168,7884,7174,6972,742,7895,6108,744,3790,7510,7704,750,7739,7913,6976,7940,762,7187,6980,7744,476,6985,7189,7198,7103,6144,7199,481,6987,6994,7105,6207,7215,7008,7111,6217,6254,495,7223,7132,6939,7761,7532,6270,7950,7026,7765,7227,497,7229,6285,381,794,7558,7795,7796,7027,7951,7628,7661,7247,7142,383,7143,7798,7252,805,7263,7664,7145,7148,7967,7058,7975,7780,6291,7158,3819,7042,6294,7162,376,7790,7981,7800,6352,6356,3826,7165,7997,8004,378,829,582,502,7166,7169,7046,6376,7703,446,7814,6418,7807,7268,7811,7279,7175,8007,7821,7825,1765,1820,1552,1671,1653,1543,1061,1785,1628,1784,842,1372,1814,1662,1517,1772,1767,1795,1533,1661,2123,1839,1670,1496,1687,637,1898,1715,1973,1762,1038,1393,1205,1439,1894,1813,1845,2093,1778,1830,1780,1921,2603,493,2624,1863,2205,1650,1926,514,2629,2268,2279,2053,1946,1677,1170,607,1500,2002,1208,2086,2058,2144,1673,1971,2168,1703,1260,1807,1329,1902,1668,1269,2159,596,2305,1443,503,2114,680,687,1840,510,7057,633,537,405,431,7183,2628,528,684,1506,2481,611,769,1435,772,7829,1476,529,546,1797,1623,1786,2173,1801,2202,2228,690,775,3187,783,2653,691,2489,539,406,1467,2696,540,817,547,659,2248,567,7089,621,1895,7055,441,411,448,699,711,2364,3837,722,7719,3258,557,2327,2553,436,1986,717,732,720,1947,565,1721,1440,2025,2670,2288,857,2182,1955,6419,6443,445,397,8012,8014,856,2021,2012,776,757,2805,2652,2582,2298,2257,3472,2375,2041,2405,1865,735,7138,459,7830,1755,763,515,2680,743,487,2613,2644,2742,403,774,780,2703,3225,666,798,462,2708,902,1692,760,2506,1976,2564,2172,1724,591,1109,875,3563,475,516,2893,803,3555,1143,853,784,610,492,880,618,2709,2845,1695,789,6445,822,3592,838,1732,3581,2573,6468,7059,2969,3639,813,512,2856,823,868,622,668,3691,673,888,2876,1974,884,513,2578,2636,3129,635,3078,4746,916,1215,3702,2637,530,4005,899,881,3899,2296,1315,560,883,4027,3316,527,2347,566,571,3508,4062,2610,6537,692,2619,3628,4069,409,906,1972,662,2638,6546,3522,2851,918,3314,2583,2427,3337,2635,2005,1907,2659,2163,3636,4850,2648,2678,2223,2206,3677,2640,1833,2124,1796,2662,2422,2186,3410,2409,1875,925,997,2187,1716,2443,2681,1118,2673,3579,2654,3596,2447,889,2651,2677,2722,1335,679,3681,1121,2261,4141,1468,2773,533,1340,4036,1491,2559,2218,3511,2580,3617,4263,2688,1572,1538,2691,2251,4068,2461,2753,2743,2588,2474,1933,2325,2647,1678,2668,1961,2332,7834,2612,1783,2343,2761,7092,2009,3709,4277,2306,2052,2432,531,1193,2683,2501,2056,4415,558,2555,1968,681,3473,2645,2078,2810,2960,4315,2818,2872,3020,2362,3575,1347,1834,2378,1528,1802,1736,1789,1808,3640,2689,2993,3922,1880,3204,3186,3367,7094,2095,3991,3235,1823,636,2063,1225,2403,2658,3368,2344,2482,1579,4432,1878,1349,4558,2436,1558,1739,2365,1581,4328,3409,2650,1674,2686,3427,892,2715,3651,3464,2433,1876,4024,3689,2507,1606,3402,2788,1746,3514,1047,2620,648,2724,1995,2077,1544,2132,3604,2527,1881,1748,3438,733,4599,1112,3440,737,1654,748,3710,2633,1806,4336,7186,2862,1720,3559,2693,1663,2774,3495,2627,2487,2526,1996,2682,4034,2566,2026,3580,1770,2459,1850,1908,1884,2924,2147,3498,1561,2684,1576,2957,2080,2674,1901,2090,2622,2701,2695,1679,2704,1699,1774,2720,2042,2727,2661,3602,1803,2730,3515,2463,3610,2734,2054,1825,1805,2741,2064,2120,1123,2131,1836,1867,2138,1827,7275,7193,1890,3984,3635,3578,4181,778,2569,3600,3985,4262,3609,702,4017,4031,1782,2750,2778,7218,3625,2976,4296,2751,1464,3621,2706,2711,4300,3442,1985,2102,3645,2122,3630,654,2198,2213,1930,2581,2784,3620,4678,655,3097,3626,3692,2718,3642,2672,3668,3700,3654,2199,3679,3632,2598,2250,2615,4867,8015,2313,3711,4307,2796,4421,3492,3676,1842,4874,4349,4858,3652,3712,3494,3714,3723,3672,1906,3869,4087,4065,4907,4066,4917,3980,2809,3982,2917,2322,2939,4028,3527,4366,3673,4885,3029,7287,3126,4021,3634,4085,3685,3133,2795,7222,4901,3425,1932,2685,4033,4040,3480,4030,4132,2855,3713,4097,2397,3718,3551,3638,3623,4114,2835,4892,1927,2200,4927,2000,1948,3612,3624,1150,1987,3633,2272,2623,4142,2277,8017,4959,7289,4193,4261,2388,2049,3902,4934,4943,4093,2066,2406,3724,563,5075,2667,3560,2062,3730,2951,2087,4213,3561,4987,3660,3976,685,3993,4981,3657,2255,580,3687,2133,2698,5006,4259,4269,3705,3733,2050,2710,4098,3994,3986,3999,786,1122,2438,4310,2744,4018,719,5019,4002,1960,3583,2848,4319,724,2787,790,4282,4431,2439,3655,2965,3852,1877,5024,4153,5027,4451,2853,2825,2829,3069,4016,2442,2106,1175,4006,5023,3116,3128,4267,4025,4007,2861,2968,5040,2162,4043,4057,4457,4332,4081,2069,4462,4252,414,3690,693,4100,4143,1510,3697,4045,544,2161,3939,4255,3190,3995,738,4103,5091,4256,2029,4324,4029,2259,1719,3152,3222,1244,4039,2470,5097,4022,4281,4338,2174,2194,3181,2368,2244,2514,1859,2492,2329,2540,2891,2091]
	qids_valid = set(qids_valid)
        print 'Seperating training into smaller training/validation sets'
        qid = 0
        for line in file:
            new_qid = int(line.split('qid:')[1].split(' ')[0])
            if qid != new_qid:
                print '...reading query %i\r' % new_qid,
            qid = new_qid
            if qid in qids_valid:
                valid_file.write(line)
            else:
                train_file.write(line)
        train_file.close()
        valid_file.close()

        #file = open(train_file)
        #n_queries = 0
        #small_train_file = os.path.join(dir_path, 'set1.small_train.txt')
        #small_valid_file = os.path.join(dir_path, 'set1.small_valid.txt')
        #train_file = open(small_train_file,'w')
        #valid_file = open(small_valid_file,'w')
        #qid_split = 13944
        #print 'Seperating training into smaller training/validation sets'
        #qid = 0
        #for line in file:
        #    new_qid = int(line.split('qid:')[1].split(' ')[0])
        #    if qid != new_qid:
        #        print '...reading query %i\r' % new_qid,
        #    qid = new_qid
        #    if qid <= qid_split:
        #        train_file.write(line)
        #    else:
        #        valid_file.write(line)
        #train_file.close()
        #valid_file.close()

        print 'Done                     '
    except IOError:
        print 'Go to http://learningtorankchallenge.yahoo.com/ to download the data.'
        print 'Once this is done, call this function again.'
