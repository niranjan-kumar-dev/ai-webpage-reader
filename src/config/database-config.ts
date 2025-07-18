import { DistanceStrategy } from "@langchain/community/vectorstores/pgvector";
import { PoolConfig } from "pg";

export default function getPgVectorStoreConfig(): {
  postgresConnectionOptions: PoolConfig;
  tableName: string;
  columns: {
    vectorColumnName: string;
    contentColumnName: string;
    metadataColumnName: string;
  };
  distanceStrategy: DistanceStrategy;
} {
  return {
    postgresConnectionOptions: {
      type: "postgres",
      host: process.env.PG_HOST || "localhost",
      port: process.env.PG_PORT ? parseInt(process.env.PG_PORT) : 5432,
      user: process.env.PG_USER || "postgres",
      password: process.env.PG_PASSWORD || "root",
      database: process.env.PG_DATABASE || "gmrt_webpages",
    } as PoolConfig,
    tableName: process.env.PG_TABLE || "embeddings",
    columns: {
      vectorColumnName: process.env.PG_VECTOR_COLUMN || "embedding",
      contentColumnName: process.env.PG_CONTENT_COLUMN || "content",
      metadataColumnName: process.env.PG_METADATA_COLUMN || "metadata",
    },
    distanceStrategy:
      (process.env.PG_DISTANCE_STRATEGY as DistanceStrategy) || "cosine",
  };
}
