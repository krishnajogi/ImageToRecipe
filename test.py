import pandas as pd
import numpy as np
import os
import requests
import cv2
from Recipe import *
import ast

recipe_list = []

index = 1001

dataset = pd.read_csv('dataset/core-data_recipe.csv')
for i in range(0,len(dataset)):
    r_id = dataset.get_value(i, 'recipe_id')
    r_name = dataset.get_value(i, 'recipe_name')
    ingredients = dataset.get_value(i, 'ingredients')
    nutritions = dataset.get_value(i, 'nutritions')
    cooking = ast.literal_eval(dataset.get_value(i, 'cooking_directions')).get('directions')
    r_name = r_name.strip().lower()
    if 'indian' in r_name:
        obj = Recipe()
        obj.setRecipeID(r_id)
        obj.setName(r_name)
        obj.setIngredients(ingredients)
        obj.setNutritions(nutritions)
        obj.setCooking(cooking)
        recipe_list.append(obj)
        url = dataset.get_value(i, 'image_url')
        os.mkdir('IndianFood/'+str(index))
        if index != 1039 and index != 1102:
            name = r_name.replace(" ","_")
            with open('IndianFood/'+str(index)+'/'+name+'.jpg', 'wb') as f:
                f.write(requests.get(url).content)
            img = cv2.imread('IndianFood/'+str(index)+'/'+name+'.jpg')
            for j in range(1,5):
                cv2.imwrite('IndianFood/'+str(index)+'/'+str(j)+'.jpg',img)
        else:
            with open('IndianFood/'+str(index)+'/'+'0.jpg', 'wb') as f:
                f.write(requests.get(url).content)
            img = cv2.imread('IndianFood/'+str(index)+'/'+'0.jpg')
            for j in range(1,5):
                cv2.imwrite('IndianFood/'+str(index)+'/'+str(j)+'.jpg',img)
        index = index + 1    
        #f.close()
        print(r_name+" "+str(index))

recipe_list = np.asarray(recipe_list)
np.save("index.txt",recipe_list)

'''
for i in range(0,len(dataset)):
    name = dataset.get_value(i, 'recipe_name')
    name = name.lower()
    if 'indian' in name:
        row = dataset.iloc[[i]]
        row = row.values
        data.append(i)
        print(row)

data = np.asarray(data)
np.save("index.txt",data)
        



index = 0

dataset = pd.read_csv('dataset/core-data_recipe.csv',nrows=1000)
for i in range(0,len(dataset)):
    url = dataset.get_value(i, 'image_url')
    os.mkdir('images/'+str(index))
    with open('images/'+str(index)+'/'+'0.jpg', 'wb') as f:
        f.write(requests.get(url).content)
    img = cv2.imread('images/'+str(index)+'/'+'0.jpg')
    for j in range(1,5):
        cv2.imwrite('images/'+str(index)+'/'+str(j)+'.jpg',img)
    index = index + 1    
    f.close()
    print(index)
    
'''
