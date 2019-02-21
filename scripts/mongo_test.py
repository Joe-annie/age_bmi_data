import pymongo
import numpy as np
import cv2
# from predictor import predict_age
import pandas as pd
import json

def connect(col_name='image', db_name='face_new', uri='mongodb://192.168.3.200:27019'):
    client = pymongo.MongoClient(uri)
    db = client.get_database(db_name)
    return db.get_collection(col_name)


# def pre_process_from_buf(buf, face_size=64):
#     try:
#         img = cv2.imdecode(np.asarray(bytearray(buf), dtype=np.uint8), -1) #转换二进制数据
#         face = cv2.resize(img, (face_size, face_size))
#         face = np.expand_dims(face, axis=0)
#         return {
#             'code': 0,
#             'face': face
#         }
#     except:
#         return {
#             'code': -1
#         }


# def batch_update():
#     col = connect('metric')
#     col_image = connect('image')
#     for c in col.find({}, {'md5': 1}):
#         md5 = c.get('md5')
#         a_dict = col_image.find_one({'md5': md5}, {'face_data': 1})
#         if a_dict:
#             face_data = a_dict.get('face_data')
#             if face_data:
#                 r = pre_process_from_buf(face_data) #转换
#                 if r['code'] == 0:
#                     face = r['face']
#                     age_result = predict_age(face) #预测
#                     print(age_result)
#                     if age_result['code'] == 0:
#                         age_predict = age_result['age']
#                         col.update_one({'md5': md5}, {'$set': {'age_predict': age_predict}})
#                         print('update {} and set age_predict to {:d}'.format(md5, age_predict))

# def main():
#     batch_update()

def getData_age():
    col = connect('metric')

    origin_list = []
    predict_list = []
    for c in col.find({'age_predict': {'$exists': True}, 'bmi_predict': {'$exists': True}},{'age': 1,'age_predict':1}):
        origin = c.get("age")
        predict = c.get('age_predict')
        origin_list.append(origin)
        predict_list.append(predict)
    dic = {'origin_age':origin_list, 'predict_age' : predict_list}
    df = pd.DataFrame(dic)
    df.to_csv("age_compare.csv",index=False)

def getData_bmi():
    col = connect('metric')

    origin_list = []
    predict_list = []
    for c in col.find({'bmi': {'$exists': True}, 'bmi_predict': {'$exists': True}},{'bmi': 1,'bmi_predict':1}):
        if c.get('bmi_predict'):
            origin = round(c.get("bmi"))
            predict = round(c.get('bmi_predict'))
            origin_list.append(origin)
            predict_list.append(predict)
    dic = {'origin_bmi':origin_list, 'predict_bmi' : predict_list}
    df = pd.DataFrame(dic)
    df.to_csv("bmi_compare.csv",index=False)
def getData_bmi_5():
    col = connect('metric')

    for tag in ['2','3','4','5','_refine']:
        origin_list_f = []
        predict_list_f = []
        origin_list_m = []
        predict_list_m = []

        for c in col.find({'bmi': {'$exists': True}, 'bmi_predict'+tag: {'$exists': True},
                           'gender':  {'$exists': True}, 'gender_mark':  {'$exists': True}},
                          {'bmi': 1,'bmi_predict'+tag:1,
                           'gender': 1, 'gender_mark': 1}):
            if c.get('gender_mark') == 0:
                gender = c.get('gender')
                if gender  == 1:
                    origin_f = round(c.get("bmi"))
                    predict_f = round(c.get('bmi_predict'+tag))
                    origin_list_f.append(origin_f)
                    predict_list_f.append(predict_f)
                elif gender == 0:
                    origin_m = round(c.get("bmi"))
                    predict_m = round(c.get('bmi_predict'+tag))
                    origin_list_m.append(origin_m)
                    predict_list_m.append(predict_m)

        dic_f = {'origin_bmi':origin_list_f, 'predict_bmi' : predict_list_f}
        df_f = pd.DataFrame(dic_f)
        df_f.to_csv("bmi"+tag+"_female"+"_compare.csv",index=False)

        dic_m = {'origin_bmi':origin_list_m, 'predict_bmi' : predict_list_m}
        df_m = pd.DataFrame(dic_m)
        df_m.to_csv("bmi"+tag+"_male"+"_compare.csv",index=False)



def hisgram_age():
    df = pd.read_csv('age_compare.csv')

    import matplotlib.pyplot as plt

    series_origin = df.groupby('origin_age').origin_age.count()
    ages = list(series_origin.index)
    count_origin = list(series_origin.values)

    series_predict = df.groupby('predict_age').predict_age.count()
    ages_predict = list(series_predict.index)
    count_predict = list(series_predict.values)

    # plt.plot(ages, count, color="red", label="Naive version of insertion sort")
    # # plt.plot(nValues, tValues, color="blue", label="Less naive version of insertion sort")
    # # plt.plot(nValuesMerge, tValuesMerge, color="orange", label="Not very slick implementation of mergeSort")
    # plt.xlabel("age")
    # plt.ylabel("number")
    # plt.legend()
    # plt.title("All sorts of sorts")
    # plt.show()

    # fig = plt.figure(figsize=(8,6))
    # df.groupby('predict_age').origin_age.count().plot.bar(ylim=0)
    # plt.show()

    # from pyecharts import Bar
    # attr1 = ages
    # attr2 = ages_predict
    # v1 = count_origin
    # v2 = count_predict
    # bar = Bar("年龄堆叠柱状图")
    # bar.add("原始年龄", attr1, v1, is_stack=True) #mark_point=["average"],
    # bar.add("预测年龄", attr2, v2, is_stack=True)  #mark_line=["min", "max"]
    # bar.render("age_bar.html")

    dic = {'age_origin':ages, 'count_origin' : count_origin}
    dic2 = {'age_predict':ages_predict,'count_predict':count_predict}
    df = pd.DataFrame(dic)
    df2 = pd.DataFrame(dic2)
    df.to_csv("age_origin_count.csv",index=False)
    df2.to_csv("age_predict_count.csv",index=False)





def hisgram_bmi():
    df = pd.read_csv('bmi_compare.csv')

    import matplotlib.pyplot as plt

    series_origin = df.groupby('origin_bmi').origin_bmi.count()
    bmi = list(series_origin.index)
    count_origin = list(series_origin.values)

    series_predict = df.groupby('predict_bmi').predict_bmi.count()
    bmi_predict = list(series_predict.index)
    count_predict = list(series_predict.values)

    # plt.plot(ages, count, color="red", label="Naive version of insertion sort")
    # # plt.plot(nValues, tValues, color="blue", label="Less naive version of insertion sort")
    # # plt.plot(nValuesMerge, tValuesMerge, color="orange", label="Not very slick implementation of mergeSort")
    # plt.xlabel("age")
    # plt.ylabel("number")
    # plt.legend()
    # plt.title("All sorts of sorts")
    # plt.show()

    # fig = plt.figure(figsize=(8,6))
    # df.groupby('predict_age').origin_age.count().plot.bar(ylim=0)
    # plt.show()

    # from pyecharts import Bar
    # attr1 = bmi
    # attr2 = bmi_predict
    # v1 = count_origin
    # v2 = count_predict
    # bar = Bar("bmi堆叠柱状图")
    # bar.add("原始bmi", attr1, v1, is_stack=True) #mark_point=["average"],
    # bar.add("预测bmi", attr2, v2, is_stack=True)  #mark_line=["min", "max"]
    # bar.render("bmi_bar.html")

    dic = {'bmi_origin':bmi, 'count_origin' : count_origin}
    dic2 = {'bmi_predict':bmi_predict,'count_predict':count_predict}
    df = pd.DataFrame(dic)
    df2 = pd.DataFrame(dic2)
    df.to_csv("bmi_origin_count.csv",index=False)
    df2.to_csv("bmi_predit_count.csv",index=False)


def hisgram_bmi_5():
    for tag in ['2','3','4','5','_refine']:
        for gender in ['_female','_male']:
            df = pd.read_csv('bmi'+tag+gender+'_compare.csv')

            series_origin = df.groupby('origin_bmi').origin_bmi.count()
            bmi = list(series_origin.index)
            count_origin = list(series_origin.values)

            series_predict = df.groupby('predict_bmi').predict_bmi.count() #data.icol(1)
            bmi_predict = list(series_predict.index)
            count_predict = list(series_predict.values)

            from pyecharts import Bar
            attr1 = bmi
            attr2 = bmi_predict
            v1 = count_origin
            v2 = count_predict
            bar = Bar("bmi"+tag+gender+"堆叠柱状图")
            bar.add("原始bmi", attr1, v1, is_stack=True) #mark_point=["average"],
            bar.add("预测bmi"+tag+gender, attr2, v2, is_stack=True)  #mark_line=["min", "max"]
            bar.render("bmi"+tag+gender+"_bar.html")

    # dic = {'bmi_origin':bmi, 'count_origin' : count_origin}
    # dic2 = {'bmi_predict':bmi_predict,'count_predict':count_predict}
    # df = pd.DataFrame(dic)
    # df2 = pd.DataFrame(dic2)
    # df.to_csv("bmi_origin_count.csv",index=False)
    # df2.to_csv("bmi_predit_count.csv",index=False)






def mark_age():
    col = connect('metric')
    for c in col.find({'age_predict': {'$exists': True}, 'bmi_predict': {'$exists': True}},
                      {'age': 1,'age_predict':1,'md5': 1}):
        origin = c.get("age")
        predict = c.get('age_predict')
        md5 = c.get('md5')
        differ = abs(predict - origin)
        mark = 0
        std = 8
        if origin <= 0 or origin >= 100:
            mark = -1
        elif differ >= std and differ < 2 * std:
            mark = 1
        elif differ >= 2 * std and differ < 3 * std:
            mark = 2
        elif differ >= 3 * std and differ < 4 * std:
            mark = 3
        elif differ >= 4 * std:
            mark = 4
        r = col.update_one({'md5': md5}, {'$set': {'age_mark': mark}})
        # print('update {} and set age_predict to {:d}'.format(md5, age_predict))
        # print(md5, mark, r.matched_count, r.modified_count)
def mark_bmi():
    col = connect('metric')
    for c in col.find({'age_predict': {'$exists': True}, 'bmi_predict': {'$exists': True}},
                      {'bmi': 1,'bmi_predict':1,'md5': 1}):
        origin = c.get("bmi")
        predict = c.get('bmi_predict')
        md5 = c.get('md5')

        minus = -1 if origin < predict else 1
        differ = abs(predict - origin)
        mark = 0
        std = 8
        if origin <= 0 or origin >= 100:
            mark = -1
        elif differ >= std and differ < 2 * std:
            mark = 1
        elif differ >= 2 * std and differ < 3 * std:
            mark = 2
        elif differ >= 3 * std and differ < 4 * std:
            mark = 3
        elif differ >= 4 * std:
            mark = 4

        r = col.update_one({'md5': md5}, {'$set': {'bmi_mark': mark}})
        # print('update {} and set age_predict to {:d}'.format(md5, age_predict))
        # print(md5, mark, r.matched_count, r.modified_count)
def mark_gender():
    col = connect('metric')
    for c in col.find({'age_predict': {'$exists': True}, 'bmi_predict': {'$exists': True}, 'gender_predict': {'$exists': True}},
                      {'gender': 1,'gender_predict':1,'md5': 1}):
        origin = c.get("gender")
        predict = c.get('gender_predict')
        md5 = c.get('md5')

        if origin == predict:
            mark = 0
        elif origin == 1 and predict == 0:
            mark = -1
        elif origin == 0 and predict == 1:
            mark = 1
        r = col.update_one({'md5': md5}, {'$set': {'gender_mark': mark}})
        # print('update {} and set age_predict to {:d}'.format(md5, age_predict))
        print(md5, mark, r.matched_count, r.modified_count)


if __name__ == '__main__':
    # getData_bmi()
    # getData_age()

    # hisgram_age()
    # hisgram_bmi()

    # df = pd.read_csv('age_compare.csv')
    #
    # series_origin = df.groupby('origin_age').origin_age.count()
    # ages = list(series_origin.index)
    # count_origin = list(series_origin.values)
    #
    # series_predict = df.groupby('predict_age').predict_age.count()
    # ages_predict = list(series_predict.index)
    # count_predict = list(series_predict.values)
    #
    # for i,num in enumerate(ages):
    #     if num == 25:
    #         print(count_predict[i])
    #         print(count_origin[i])
    #         break
    # m = 0
    # for num in count_predict:
    #     m += num
    # print('predict_agr',m)
    # print(df.describe())
    # getData_bmi_5()
    hisgram_bmi_5()





    

