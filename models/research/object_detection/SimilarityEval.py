from Similarity import *
score1=0
score2=0
score3=0
score4=0
scores=[score1,score2,score3,score4]

def getTheMostSimilar(similarityList,theCorrectPath,accuracyDict,modelType,printOut):
    minValue_0=float('inf')
    index_0=None

    minValue_1=float('inf')
    index_1=None

    minValue_2=float('inf')
    index_2=None

    minValue_3=float('inf')
    index_3=None

    for indexNbr,cases in enumerate(similarityList):
    	# 	Loop over the possible outfits
        #	print("The image is:{} and the distances is:{}".format(indexNbr+1,cases))
        if cases[0] < minValue_0:
            index_0=indexNbr
            minValue_0=cases[0]

        if cases[1] < minValue_1:
            index_1=indexNbr
            minValue_1=cases[1]

        if cases[2] < minValue_2:
            index_2=indexNbr
            minValue_2=cases[2]
        if cases[3] < minValue_3:
            index_3=indexNbr
            minValue_3=cases[3]
    s = ''.join(x for x in theCorrectPath[0] if x.isdigit())
    # Here is the problem
    # Needs to get the last number form the string
    # Make a specific regex expression! 
    # Solve tomorrow! 
    
    theCorrectIndex=int(s[-1])
    indexes=[index_0,index_1,index_2,index_3]
    #embed()
    #for index in enumerate(indexes):
    #	if (index+1)==theCorrectIndex:
    #		scores[index]=scores[index]+1

    if (index_0+1)==theCorrectIndex:
    	accuracyDict[model][0]=accuracyDict[model][0]+1
    elif (index_1+1)==theCorrectIndex:
    	accuracyDict[model][1]=accuracyDict[model][1]+1
    elif (index_2+1)==theCorrectIndex:
    	accuracyDict[model][2]=accuracyDict[model][2]+1
    elif (index_3+1)==theCorrectIndex:
    	accuracyDict[model][3]=accuracyDict[model][3]+1

    if(printOut):
    	print("The most similar image based measurment {} is HM{}.jpg the correct image is {}".format(1, index_0+1,theCorrectIndex))
    	print("The most similar image based measurment {} is HM{}.jpg the correct image is {}".format(2, index_1+1,theCorrectIndex))
    	print("The most similar image based measurment {} is HM{}.jpg the correct image is {}".format(3, index_2+1,theCorrectIndex))
    	print("The most similar image based measurment {} is HM{}.jpg the correct image is {}".format(4, index_3+1,theCorrectIndex))
 
    return (minValue_1,index_1,accuracyDict)


EXTRACTED_IMAGE_PATH=['PicturesTest2/out6.jpg']
#print("The correct image is " + EXTRACTED_IMAGE_PATH[0])
clothsmyPathList=['PicturesTest2/HM1.jpg','PicturesTest2/HM2.jpg','PicturesTest2/HM3.jpg','PicturesTest2/HM4.jpg','PicturesTest2/HM5.jpg','PicturesTest2/HM6.jpg','PicturesTest2/HM7.jpg','PicturesTest2/HM8.jpg','PicturesTest2/HM9.jpg','PicturesTest2/HM10.jpg','PicturesTest2/HM11.jpg','PicturesTest2/HM12.jpg']
# similarityModel=BuildBase("Deep")
# similarityModel.setImagePathCloths(clothsmyPathList)
# similarityModel.setImageExtracted(EXTRACTED_IMAGE_PATH)
# minValue,index=getTheMostSimilar(similarityModel.getSimilaririesForList())

# similarityModel2=BuildBase("Hist")
# similarityModel2.setImagePathCloths(clothsmyPathList)
# similarityModel2.setImageExtracted(EXTRACTED_IMAGE_PATH)
# minValue,index=getTheMostSimilar(similarityModel2.getSimilaririesForList())

# similarityModel2=BuildBase("Deep2")
# similarityModel2.setImagePathCloths(clothsmyPathList)
# similarityModel2.setImageExtracted(EXTRACTED_IMAGE_PATH)
# minValue,index=getTheMostSimilar(similarityModel2.getSimilaririesForList())



EXTRACTED_IMAGE_PATH_LIST=[['PicturesTest2/out1.jpg'],['PicturesTest2/out2.jpg'],['PicturesTest2/out5.jpg'],
							['PicturesTest2/out8.jpg'],['PicturesTest2/out9.jpg'],['PicturesTest2/out10.jpg'],
							['PicturesTest2/out11.jpg'],['PicturesTest2/out12.jpg'],['PicturesTest2/out13.jpg']]
modelTypes=["Deep","Hist","Deep2"]
accuracyDict = {'Deep': [0,0,0,0], 'Hist': [0,0,0,0], 'Deep2': [0,0,0,0]}
for EXTRACTED_IMAGE_PATH in EXTRACTED_IMAGE_PATH_LIST:
	print("The correct image is " + EXTRACTED_IMAGE_PATH[0])
	for model in modelTypes:
		similarityModel=BuildBase(model)
		similarityModel.setImagePathCloths(clothsmyPathList)
		similarityModel.setImageExtracted(EXTRACTED_IMAGE_PATH)
		minValue,index,accuracyList=getTheMostSimilar(similarityModel.getSimilaririesForList(),EXTRACTED_IMAGE_PATH,accuracyDict,model,True)
print(accuracyDict)
