%1. Read all the images in the train/val folders
%2. Save the path in the image_file.lua
%3. Save the label_id which is the id when read in alpha order
mode = 'val'; %val;
image_file = fopen('image_gt.lua','w');
label_file = fopen('label_gt.lua','w');
fprintf(image_file,'return{');
fprintf(label_file,'return{');

mainpath = strcat('/home/sjvision/spinoza/SaumyaJetley/ILSVRC/trainval/',mode);
folders = dir(mainpath);
for i=1:size(folders,1)
    if(strcmp(folders(i).name,'.') || strcmp(folders(i).name,'..'))
        continue
    else
        files = dir(strcat(mainpath, '/', folders(i).name,'/*.JPEG'));
        for j=1:size(files,1)
            fprintf(image_file,'\"%s/%s/%s\",\n',mainpath, folders(i).name, files(j).name);
            fprintf(label_file,'\"%d\",\n',(i-2));
        end
    end
end

fprintf(image_file,'}');
fprintf(label_file,'}');
