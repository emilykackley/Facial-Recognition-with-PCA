%--------------------------------Method 1--------------------------------%
%For each subject, use the first five images (1.pgm to 5.pgm) for training 
%the subspace. Use the files 6.pgm to 10.pgm for the performance evaluation

%Directories of all 40 subjects
S1 = dir('\\kc.umkc.edu\kc-users\home\e\ek6w8\My Documents\Project #1\att_faces\s1\*.pgm')
S2 = dir('\\kc.umkc.edu\kc-users\home\e\ek6w8\My Documents\Project #1\att_faces\s2\*.pgm')
S3 = dir('\\kc.umkc.edu\kc-users\home\e\ek6w8\My Documents\Project #1\att_faces\s3\*.pgm')
S4 = dir('\\kc.umkc.edu\kc-users\home\e\ek6w8\My Documents\Project #1\att_faces\s4\*.pgm')
S5 = dir('\\kc.umkc.edu\kc-users\home\e\ek6w8\My Documents\Project #1\att_faces\s5\*.pgm')
S6 = dir('\\kc.umkc.edu\kc-users\home\e\ek6w8\My Documents\Project #1\att_faces\s6\*.pgm')
S7 = dir('\\kc.umkc.edu\kc-users\home\e\ek6w8\My Documents\Project #1\att_faces\s7\*.pgm')
S8 = dir('\\kc.umkc.edu\kc-users\home\e\ek6w8\My Documents\Project #1\att_faces\s8\*.pgm')
S9 = dir('\\kc.umkc.edu\kc-users\home\e\ek6w8\My Documents\Project #1\att_faces\s9\*.pgm')
S10 = dir('\\kc.umkc.edu\kc-users\home\e\ek6w8\My Documents\Project #1\att_faces\s10\*.pgm')
S11 = dir('\\kc.umkc.edu\kc-users\home\e\ek6w8\My Documents\Project #1\att_faces\s11\*.pgm')
S12 = dir('\\kc.umkc.edu\kc-users\home\e\ek6w8\My Documents\Project #1\att_faces\s12\*.pgm')
S13 = dir('\\kc.umkc.edu\kc-users\home\e\ek6w8\My Documents\Project #1\att_faces\s13\*.pgm')
S14 = dir('\\kc.umkc.edu\kc-users\home\e\ek6w8\My Documents\Project #1\att_faces\s14\*.pgm')
S15 = dir('\\kc.umkc.edu\kc-users\home\e\ek6w8\My Documents\Project #1\att_faces\s15\*.pgm')
S16 = dir('\\kc.umkc.edu\kc-users\home\e\ek6w8\My Documents\Project #1\att_faces\s16\*.pgm')
S17 = dir('\\kc.umkc.edu\kc-users\home\e\ek6w8\My Documents\Project #1\att_faces\s17\*.pgm')
S18 = dir('\\kc.umkc.edu\kc-users\home\e\ek6w8\My Documents\Project #1\att_faces\s18\*.pgm')
S19 = dir('\\kc.umkc.edu\kc-users\home\e\ek6w8\My Documents\Project #1\att_faces\s19\*.pgm')
S20 = dir('\\kc.umkc.edu\kc-users\home\e\ek6w8\My Documents\Project #1\att_faces\s20\*.pgm')
S21 = dir('\\kc.umkc.edu\kc-users\home\e\ek6w8\My Documents\Project #1\att_faces\s21\*.pgm')
S22 = dir('\\kc.umkc.edu\kc-users\home\e\ek6w8\My Documents\Project #1\att_faces\s22\*.pgm')
S23 = dir('\\kc.umkc.edu\kc-users\home\e\ek6w8\My Documents\Project #1\att_faces\s23\*.pgm')
S24 = dir('\\kc.umkc.edu\kc-users\home\e\ek6w8\My Documents\Project #1\att_faces\s24\*.pgm')
S25 = dir('\\kc.umkc.edu\kc-users\home\e\ek6w8\My Documents\Project #1\att_faces\s25\*.pgm')
S26 = dir('\\kc.umkc.edu\kc-users\home\e\ek6w8\My Documents\Project #1\att_faces\s26\*.pgm')
S27 = dir('\\kc.umkc.edu\kc-users\home\e\ek6w8\My Documents\Project #1\att_faces\s27\*.pgm')
S28 = dir('\\kc.umkc.edu\kc-users\home\e\ek6w8\My Documents\Project #1\att_faces\s28\*.pgm')
S29 = dir('\\kc.umkc.edu\kc-users\home\e\ek6w8\My Documents\Project #1\att_faces\s29\*.pgm')
S30 = dir('\\kc.umkc.edu\kc-users\home\e\ek6w8\My Documents\Project #1\att_faces\s30\*.pgm')
S31 = dir('\\kc.umkc.edu\kc-users\home\e\ek6w8\My Documents\Project #1\att_faces\s31\*.pgm')
S32 = dir('\\kc.umkc.edu\kc-users\home\e\ek6w8\My Documents\Project #1\att_faces\s32\*.pgm')
S33 = dir('\\kc.umkc.edu\kc-users\home\e\ek6w8\My Documents\Project #1\att_faces\s33\*.pgm')
S34 = dir('\\kc.umkc.edu\kc-users\home\e\ek6w8\My Documents\Project #1\att_faces\s34\*.pgm')
S35 = dir('\\kc.umkc.edu\kc-users\home\e\ek6w8\My Documents\Project #1\att_faces\s35\*.pgm')
S36 = dir('\\kc.umkc.edu\kc-users\home\e\ek6w8\My Documents\Project #1\att_faces\s36\*.pgm')
S37 = dir('\\kc.umkc.edu\kc-users\home\e\ek6w8\My Documents\Project #1\att_faces\s37\*.pgm')
S38 = dir('\\kc.umkc.edu\kc-users\home\e\ek6w8\My Documents\Project #1\att_faces\s38\*.pgm')
S39 = dir('\\kc.umkc.edu\kc-users\home\e\ek6w8\My Documents\Project #1\att_faces\s39\*.pgm')
S40 = dir('\\kc.umkc.edu\kc-users\home\e\ek6w8\My Documents\Project #1\att_faces\s40\*.pgm')

%Create filepaths for all subjects
filepaths = [];
for i = 1: 40
    a = int2str(i)
    x = strcat('\\kc.umkc.edu\kc-users\home\e\ek6w8\My Documents\Project #1\att_faces\s',a,'\');
    filepaths{i}=x
end
%Create directories (string) of all subject folders
sub_dir = [];
for i = 1: 40
    a = int2str(i)
    x = strcat('S',a);
    x = matlab.lang.makeValidName(x)
    sub_dir{i}=S1
end

%Create a training set made of the first 5 images for each test subject
%(200 images total)
training = cell(1,200);
%Create a training set made of the first 5 images for each test subject
%(200 images total)
testing = cell(1,200);

%Read all images into the appropriate training and testing arrays
[training,testing] = sub_images(sub_dir,filepaths)

%Convert cell arrays to matrices
training = cell2mat(training);
testing = cell2mat(testing);

%Calculate the mean of the data matrix
m = mean(training,2);

%Subtract the mean from each image
d = training-repmat(m,1,200);

%Compute the covariance matrix
co = d*d';

%Calculate the eigenvalues and eigenvectors of the covariance matrix
[eigenvectors, eigenvalues] = eig(co);

%Sort the eigenvectors by eigenvalues
eigenvalues = diag(eigenvalues);
[temp,index] = sort(eigenvalues,'descend');

%Compute the number of eigenvalues greater than zero
c1 = 0;
for i = 1: size(eigenvalues,1)
    if(eigenvalues(i)>0)
        c1 = c1 + 1;
    end
end

%Use the eigenvectors that the corresponding eigenvalues that are greater
%than zero (this threshold can be changed to any value you want)
vec = eigenvectors(:,index(1:200));

%Projecting the training data
project_trainimg = vec'*d;

%Subtract the mean from each testing image
testing = testing-repmat(mean(testing,2),1,200);

%Project test images
project_testimg = vec'*testing;

%Euclidean distance 
D = pdist2(project_trainimg',project_testimg','Euclidean');

%Labels (determining what was correctly classified and what was not)
results = zeros(200,200);
for i = 1: 200
    for k = 1: 200
        if(fix((i-1)/10)==fix((k-1)/10))
            results(i,k)=0
        else
            results(i,k)=1;
        end
    end
end

%Find and plot the ROC curve
ezroc3(D,results,2,'',1);

%Function to import images from att_faces folder
function [train,test] = sub_images(sub_directory,filepath)
%200 training images (first 5 from each subject)
train = cell(1,200);
%200 testing images (last 5 from each test subject)
test = cell(1,200);
%a and b are placeholders for traversing through the train and test cell
%arrays
a = 1;
b = 1;

%Loop for all 40 subjects
for k = 1: 40
    fp = filepath{k}
    x = sub_directory{k}
    directory = x
    for i = 1: 10
        if i < 6
            %Get filename
            filename = strcat(fp,directory(i).name);
            %Read image
            temp = imread(filename);
            %Reshape image
            temp = reshape(temp,prod(size(temp)),1);
            temp = double(temp);
            %Add image to training set
            train{a}=temp;
            a = a+1;
        end
        if i>=6
            %Get filename
            filename = strcat(fp,directory(i).name);
            %Raed image
            temp = imread(filename);
            %Reshape image
            temp = reshape(temp,prod(size(temp)),1);
            temp = double(temp);
            %Add image to testing set
            test{b}=temp;
            b = b+1;
        end
    end
end
end

%Function plotting the ROC curve
function [roc,EER,area,EERthr,ALLthr,d,gen,imp]=ezroc3(H,T,plot_stat,headding,printInfo)%,rbst
t1=min(min(min(H)));
t2=max(max(max(H)));
num_subj=size(H,1);

stp=(t2-t1)/500;   %step size here is 0.2% of threshold span, can be adjusted

if stp==0   %if all inputs are the same...
    stp=0.01;   %Token value
end
ALLthr=(t1-stp):stp:(t2+stp);
if (nargin==1 || (nargin==3 &&  isempty(T))||(nargin==2 &&  isempty(T))||(nargin==4 &&  isempty(T))||(nargin==5 &&  isempty(T)))  %Using only H, multi-class case, and maybe 3D or no plot
    GAR=zeros(503,size(H,3));  %initialize for accumulation in case of multiple H (on 3rd dim of H)
    FAR=zeros(503,size(H,3));
    gen=[]; %genuine scores place holder (diagonal of H), for claculation of d'
    imp=[]; %impostor scores place holder (non-diagonal elements of H), for claculation of d'
    for setnum=1:size(H,3); %multiple H measurements (across 3rd dim, where 2D H's stack up)
        gen=[gen; diag(H(:,:,setnum))]; %digonal scores
        imp=[imp; H(find(not(eye(size(H,2)))))]; %off-diagonal scores, with off-diagonal indices being listed by find(not(eye(size(H,2)))) 
        for t=(t1-stp):stp:(t2+stp),    %Note that same threshold is used for all H's, and we increase the limits by a smidgeon to get a full curve
            ind=round((t-t1)/stp+2);   %current loop index, +2 to start from 1
            id=H(:,:,setnum)>t;
            
            True_Accept=trace(id);  %TP
            False_Reject=num_subj-True_Accept;  %FN
            % In the following, id-diag(diag(id)) simply zeros out the diagonal of id
            True_Reject=sum( sum( (id-diag(diag(id)))==0 ) )-size(id,1); %TN, number of off-diag zeros. We need to subtract out the number of diagonals, as 'id-diag(diag(id))' introduces those many extra zeros into the sum
            False_Accept=sum( sum( id-diag(diag(id)) ) ); %FP, number of off-diagonal ones
            
            GAR(ind,setnum)=GAR(ind,setnum)+True_Accept/(True_Accept+False_Reject); %1-FRR, Denum: all the positives (correctly IDed+incorrectly IDed)
            FAR(ind,setnum)=FAR(ind,setnum)+False_Accept/(True_Reject+False_Accept); %1-GRR, Denum: all the negatives (correctly IDed+incorrectly IDed)
        end
    end
    GAR=sum(GAR,2)/size(H,3);   %average across multiple H's
    FAR=sum(FAR,2)/size(H,3);
elseif (nargin==2 || nargin==3 || nargin == 4 || nargin == 5),   %Regular, 1-class-vs-rest ROC, and maybe 3D or no plot
    gen=H(find(T)); %genuine scores
    imp=H(find(not(T))); %impostor scores
    for t=(t1-stp):stp:(t2+stp),    %span the limits by a smidgeon to get a full curve
        ind=round((t-t1)/stp+2);   %current loop index, +2 to start from 1
        id=H>t;
        
        True_Accept=sum(and(id,T)); %TP
        False_Reject=sum(and(not(id),T));   %FN
        
        True_Reject=sum(and(not(id),not(T)));   %TN
        False_Accept=sum(and(id,not(T)));   %FP
        
        GAR2(ind)=True_Accept/(True_Accept+False_Reject); %1-FRR, Denum: all the positives (correctly IDed+incorrectly IDed)
        FAR2(ind)=False_Accept/(True_Reject+False_Accept); %1-GRR, Denum: all the negatives (correctly IDed+incorrectly IDed)
        
    end
    GAR=GAR2';
    FAR=FAR2';
end
roc=[GAR';FAR'];
FRR=1-GAR;
[e ind]=min(abs(FRR'-FAR'));    %This is Approx w/ error e. Fix by linear inerpolation of neigborhood and intersecting w/ y=x
EER=(FRR(ind)+FAR(ind))/2;
area=abs(trapz(roc(2,:),roc(1,:)));
EERthr=t1+(ind-1)*stp;%EER threshold

d=abs(mean(gen)-mean(imp))/(sqrt(0.5*(var(gen)+var(imp))));   %Decidability or d'

if (nargin==1 || nargin==2 || nargin==3 || nargin == 4 || nargin == 5)
    if plot_stat == 2
        if printInfo == 1
            figure, plot(roc(2,:),roc(1,:),'LineWidth',3),axis([-0.002 1 0 1.002]),title(['ROC Curve: ' headding '   EER=' num2str(EER) ',   Area=' num2str(area) ',   Decidability=' num2str(d)]),xlabel('FAR'),ylabel('GAR');
        elseif printInfo == 0
            figure, plot(roc(2,:),roc(1,:),'LineWidth',3),axis([-0.002 1 0 1.002]),title(['ROC Curve: ' headding ' ']),xlabel('FAR'),ylabel('GAR');
        end
    elseif plot_stat == 3
        if printInfo == 1
            figure, plot3(roc(2,:),roc(1,:),ALLthr,'LineWidth',3),axis([0 1 0 1 (t1-stp) (t2+stp)]),title(['3D ROC Curve: ' headding '   EER=' num2str(EER) ',   Area=' num2str(area)  ',   Decidability=' num2str(d)]),xlabel('FAR'),ylabel('GAR'),zlabel('Threshold'),grid on,axis square;
        elseif printInfo == 0
            figure, plot3(roc(2,:),roc(1,:),ALLthr,'LineWidth',3),axis([0 1 0 1 (t1-stp) (t2+stp)]),title(['3D ROC Curve: ' headding ' ']),xlabel('FAR'),ylabel('GAR'),zlabel('Threshold'),grid on,axis square;
        end     
    else
        %else it must be 0, i.e. no plot
    end
end
end