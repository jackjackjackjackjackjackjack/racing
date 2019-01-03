insert into inputdata
select race_id,headcount,order_of_finish,horse_number,grade,inf.distance
	,case when fea.surface = "��" then 1 else 0 end as surface_turf
	,case when fea.surface = "�_" then 1 else 0 end as surface_dirt
	,case when fea.weather = "��" then 1 else 0 end as weather_sunny
	,case when fea.weather = "��" then 1 else 0 end as weather_cloudy
	,case when fea.weather = "�J" then 1 else 0 end as weather_rainy
	,case when fea.weather = "���J" then 1 else 0 end as weather_littlerainy
	,case when fea.weather = "��" then 1 else 0 end as weather_else
	,case when course = "��" then 1 else 0 end as course_left
	,case when course = "�E" then 1 else 0 end as course_right
	,case when course = "����" then 1 else 0 end as course_straight
	,case when course = "�O" then 1 else 0 end as course_outer
	,case when course = "��" then 1 else 0 end as course_else
	,case when placeCode = "����" then 1 else 0 end as placeCode_tokyo
	,case when placeCode = "���R" then 1 else 0 end as placeCode_nakayama
	,case when placeCode = "���s" then 1 else 0 end as placeCode_kyoto
	,case when placeCode = "��_" then 1 else 0 end as placeCode_hanshin
	,case when placeCode = "����" then 1 else 0 end as placeCode_hakodate
	,case when placeCode = "�D�y" then 1 else 0 end as placeCode_sapporo
	,case when placeCode = "�V��" then 1 else 0 end as placeCode_niigata
	,case when placeCode = "����" then 1 else 0 end as placeCode_chukyo
	,case when placeCode = "���q" then 1 else 0 end as placeCode_kokura
	,case when placeCode = "����" then 1 else 0 end as placeCode_fukusima
	,femaleOnly
	,case when SUBSTR(surface_state,INSTR(inf.surface_state,':') + 1,3) = "��" then 1 else 0 end as surface_state_good
	,case when SUBSTR(surface_state,INSTR(inf.surface_state,':') + 1,3) = "�c�d" then 1 else 0 end as surface_state_littleheavy
	,case when SUBSTR(surface_state,INSTR(inf.surface_state,':') + 1,3) = "�d" then 1 else 0 end as surface_state_heavy
	,case when SUBSTR(surface_state,INSTR(inf.surface_state,':') + 1,3) = "�s��" then 1 else 0 end as surface_state_bad
	,race_number
	,ridingStrongJockey,age,dhweight,disRoc,enterTimes,fea.eps,hweight,odds
	,case when sex = "��" then 1 else 0 end as sex_male
	,case when sex = "��" then 1 else 0 end as sex_female
	,case when sex = "�Z" then 1 else 0 end as sex_shemale
	,weight,winRun,fea.preOOF,fea.pre2OOF,fea.preLastPhase 
	,preHeadCount,surfaceChanged,gradeChanged
	,cast(substr(finishing_time,1,1) as int)*60+cast(substr(finishing_time,3,2) as int)+cast(substr(finishing_time,6,1) as int)*0.1 as finishing_time
from feature as fea
left join race_info as inf
on fea.race_id = inf.id;