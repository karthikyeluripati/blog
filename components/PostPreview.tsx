import Link from "next/link"
import { PostMetadata } from "./PostMetadata"

const PostPreview = (props:PostMetadata)=>{
    return (
    // <div className='border border-voilet-300 p-4 rounded-md shadow-md 
    // bg-white'>
    <div className=''>
            <div className="flex justify-between items-center">
                <Link href={`/posts/${props.slug}`} className="flex-grow">
                    <h2 className="text-black-900 hover:underline hover:text-orange-700 mr-4">{props.title}</h2>
                </Link>
                <p className="text-sm text-slate-400 whitespace-nowrap">{props.date}</p>
            </div>
        </div>
    );
}
export default PostPreview