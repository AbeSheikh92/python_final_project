.PHONY: data

data: data/trained_model

data/trained_model:
	aws s3 cp s3://2020fa-final-project-bucket/data/models/ALL_CAPTIONS_CNN/trained_embedding.model data/models/ALL_CAPTIONS_CNN/ --request-payer=requester
	aws s3 cp s3://2020fa-final-project-bucket/data/models/ALL_CAPTIONS_FOX/trained_embedding.model data/models/ALL_CAPTIONS_FOX/ --request-payer=requester
