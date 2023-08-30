# If I launch an sbatch run without committing git changes my script will fail 
# immediately (because it checks for uncommitted changes for reproducibility). So
# I will use this simple script to launch jobs.

JOB="${1}"

if [[ `git status --porcelain` ]]; then
  echo "There are uncommitted changes; commit them then rerun"
  exit 1
fi

sbatch "${JOB}"
