import fs from 'fs';
import matter from "gray-matter";
import { ProjectPostMetadata } from './ProjectPostMetadata';

const getProjectPostMetadata = (): ProjectPostMetadata[] =>{
    const folder = 'projectposts/';
    const files=fs.readdirSync(folder);
    const markdownPosts = files.filter((file)=>file.endsWith(".md"));
    
    const posts = markdownPosts.map((fileName)=>{
      const fileContents=fs.readFileSync(`projectposts/${fileName}`,"utf8");
      const matterResult=matter(fileContents);
      return{
        title: matterResult.data.title,
        date: matterResult.data.date,
        subtitle:matterResult.data.subtitle,
        slug:fileName.replace('.md',""),
      };
    });
    return posts;
  };

  export default getProjectPostMetadata;