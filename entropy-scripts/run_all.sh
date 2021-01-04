IDS=$1
i=0
for gym_id in `cat "$IDS"`; do
	bash run_model_shaping.sh $(echo "m2_$i_$gym_id" | tr "/" "_") "$gym_id" "$i";
	bash run_baseline.sh $(echo "b_$i_$gym_id" | tr "/" "_") "$gym_id" "$i";
	sleep 2
	((++i))
done
