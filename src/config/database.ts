import { Pool } from "pg";
import getPgVectorStoreConfig from "./database-config";
import * as dotenv from "dotenv";
import path from "path";

dotenv.config({ path: path.join(__dirname, "../../.env") });

const config = getPgVectorStoreConfig();
const pool = new Pool(config.postgresConnectionOptions);

export default pool;
