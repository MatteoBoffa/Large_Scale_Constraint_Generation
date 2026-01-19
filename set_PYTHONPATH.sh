ENV_NAME="rule_constrainer"
env_path=$(conda env list | grep "$ENV_NAME" | awk '{print $2}')
if [[ "$env_path" == "*" ]]; then
	# Environment was installed, but is currently active
	echo 'FAILED: Deactivate your environment before running this script (`conda deactivate`)'
elif [[ -d "$env_path" ]]; then 
    	# Make sure that, when we activate the environment, the PYTHONPATH is set to the current directory
    	mkdir -p "$env_path/etc/conda/activate.d"    
   	# Use $(pwd) to get the current path correctly
   	CURRENT_PATH=$(pwd)    
   	# Create the env_vars.sh file to set PYTHONPATH
   	echo "export PYTHONPATH=\"$CURRENT_PATH\"" > "$env_path/etc/conda/activate.d/env_vars.sh"
    	echo "SUCCESS: Remember to run 'conda activate $ENV_NAME' to activate the effects in the current session!"
else
	# Environment is not installed yet
	echo "Failed! You shall run this script if you already installed the projectory conda environment!"
	echo "Read the README.md file for more instructions"
fi

