import fs from 'fs';
import Markdown from 'markdown-to-jsx';
import matter from 'gray-matter'
import getProjectPostMetadata from '../../../../components/getProjectPostMetadata';

const getProjectPostContent = (slug:string)=>{
    const folder = "projectposts/";
    const file= `${folder}${slug}.md`;
    const content = fs.readFileSync(file,"utf8");
    const matterResult=matter(content);
    return matterResult;
};

export const generateStaticParams = async () => {
    const posts=getProjectPostMetadata();
    return posts.map((post) => ({
        slug: post.slug
    }));
};

const ProjectPage = (props: any) => {
    const slug = props.params.slug;
    const post = getProjectPostContent(slug);
    return (
        <div>
            <div className='my-12 text-center'>
            <h1 className='text-2xl text-slate-600 text-center'>{post.data.title}</h1>
            <p className='text-slate-400 mt-2'>{post.data.date}</p>
            </div>
            <article className="prose">
            <Markdown>{post.content}</Markdown>
            </article>
        </div>
    );
};

export default ProjectPage;
