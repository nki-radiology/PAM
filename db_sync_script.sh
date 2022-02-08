for path in *; do
    databricks fs cp --recursive --overwrite $path dbfs:/PAM/$path
done
