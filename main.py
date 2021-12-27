import subprocess
import os
import torch

def get_feature(inputFile,save_path):
    basename = os.path.splitext(os.path.basename(inputFile))[0]
    extention = os.path.splitext(os.path.basename(inputFile))[1]

    #mp4ファイルで無ければffmpegによってmp4ファイルに変換する
    if not extention=='.mp4':
        print(extention+" -> mp4")
        subprocess.call(['bash', './convert.sh', "./inputs/"+basename+extention,"./inputs/"+basename+".mp4"])

    if torch.cuda.device_count()>1:
        subprocess.call(['bash', './command_MultiGPU.sh', inputFile,save_path])
    else:
        subprocess.call(['bash', './command_SingleGPU.sh', inputFile,save_path])

def main():
    os.makedirs("./outputs",exist_ok=True)  #出力用のファイルを設置するフォルダ
    os.makedirs("./results",exist_ok=True)  #文章生成結果用のファイルを設置するフォルダ

    videoname="women_long_jump.mp4"
    title=os.path.splitext(os.path.basename(videoname))[0]

    input=os.path.join("../../../inputs/", videoname)   #./BMT/submodules/video_features/から見たINPUTディレクトリ
    feature_save_path='../../../outputs/'+title+"/"         #./BMT/submodules/video_features/から見たOUTPUTディレクトリ

    if not os.path.exists('./outputs/'+title+"/"):
        os.makedirs('./outputs/'+title+"/")      #入力映像に対する抽出された特徴を保存するディレクトリ
        #特徴抽出開始
        get_feature(input,feature_save_path)
        print('特徴抽出 完了')
    else:
        print('特徴抽出　済')

    #文章生成開始
    from single_video_prediction import inference
    inf=inference(title,1000)
    inf()

if __name__ == '__main__':
    main()